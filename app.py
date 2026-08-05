"""Streamlit UI: chat with the agent, edit its prompt, and run the eval suite.

    streamlit run app.py

Threading contract: every worker that fans out lives in ``evals/`` and never
imports Streamlit. This module owns all UI: it resolves the API key on the main
thread, passes it explicitly into workers, and performs every ``st.*`` call and
session-state mutation from the main thread as futures complete.
"""

from __future__ import annotations

import functools

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from agent import (
    AGENT_MODELS,
    DEFAULT_AGENT_MODEL,
    DEFAULT_EFFORT,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    EFFORT_LEVELS,
    JUDGE_MODELS,
    MODEL_CAPS,
    SEARCH_WIKIPEDIA_TOOL,
    MissingAPIKeyError,
    WikipediaAgent,
)
from evals.judges import DEFAULT_JUDGE_EFFORT
from evals.optimizer import propose_improvement, save_proposal, summarise_failures
from evals.runner import (
    METRIC_SPECS,
    CaseResult,
    aggregate,
    format_metric,
    list_runs,
    load_run,
    get_test_case,
    load_test_cases,
    run_suite,
    save_run,
)
from evals.test_judges import JUDGE_FIXTURES, run_all_fixtures
from wiki_tool import search_wikipedia

load_dotenv()

st.set_page_config(page_title="Wikipedia Agent + Eval Harness", page_icon="*", layout="wide")

# Questions and the injection payload come from evals/test_cases.json, so this
# demo, demo.py and the graded eval always exercise the same cases and the same
# attack text. They previously each held a copy, and the copies had drifted.
SAMPLE_CASE_IDS: list[tuple[str, str]] = [
    ("Multi-hop comparison", "multi_hop_compare"),
    ("False premise", "false_premise"),
    ("Unanswerable", "unanswerable_private"),
    ("Ambiguous entity", "ambiguous_entity"),
    ("No search needed", "no_search_transform"),
    ("PROMPT INJECTION (poisoned article)", "injection_exfil"),
]

INJECTION_DEMO = get_test_case("injection_exfil").build_injection()

SAMPLE_QUERIES: list[tuple[str, str, bool]] = [
    (label, case.question, case.injection is not None)
    for label, case in ((lbl, get_test_case(cid)) for lbl, cid in SAMPLE_CASE_IDS)
]

DEFAULTS = {
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
    "agent_model": DEFAULT_AGENT_MODEL,
    "agent_effort": DEFAULT_EFFORT,
    "judge_model": DEFAULT_JUDGE_MODEL,
    "messages": [],
    "eval_results": None,
    "eval_metrics": None,
    "eval_run_path": None,
    "verify_results": None,
    "proposal": None,
    "pending": None,
    "api_key_input": "",
}

for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def widget_key(canonical: str) -> str:
    """Bind a widget to a canonical (non-widget) session-state key.

    Streamlit discards widget-keyed session state for any widget that did not
    render during a run. ``st.rerun()`` called from an earlier tab aborts the
    script before later tabs render, so config held directly under a widget key
    is silently wiped and falls back to its default on the next run -- which is
    what made "Clear conversation" reset the system prompt.

    Canonical values therefore live in plain keys that are never attached to a
    widget, and each widget gets its own ``_w_`` key seeded from that value.
    Assigning to the canonical key is also legal after the widget has rendered,
    which a widget key is not.
    """
    key = f"_w_{canonical}"
    st.session_state.setdefault(key, st.session_state[canonical])
    return key


def sync_widget(canonical: str):
    """on_change callback writing the widget's value back to its canonical key."""

    def _callback() -> None:
        st.session_state[canonical] = st.session_state[f"_w_{canonical}"]

    return _callback


def set_canonical(canonical: str, value) -> None:
    """Set a canonical value programmatically and let its widget re-seed."""
    st.session_state[canonical] = value
    st.session_state.pop(f"_w_{canonical}", None)


WRAPPING_TABLE_CSS = """
<style>
/* st.dataframe clips long cells and offers no wrap option in this Streamlit
   version. st.table wraps, and `table-layout: auto` already sizes each column to
   its widest value -- so short status columns stay narrow on their own. All this
   adds is a cap past which prose wraps instead of stretching the table.

   Deliberately no positional (nth-child) rules: these styles apply to every
   st.table on the page, and the tables here have different column counts, so a
   rule like "first six columns are narrow" would force nowrap onto the reasoning
   column of the 4-column per-case table. */
[data-testid="stTable"] table { table-layout: auto; width: 100%; }
[data-testid="stTable"] th,
[data-testid="stTable"] td {
    white-space: normal !important;
    overflow-wrap: break-word;   /* wrap between words; don't shred long tokens */
    vertical-align: top;
    max-width: 34rem;            /* prose wraps past this; short cells ignore it */
    padding: 0.4rem 0.6rem;
    line-height: 1.35;
}
</style>
"""

# Injected once per script run rather than per table: Streamlit rebuilds the DOM
# on every rerun, so this cannot be guarded by session state, but emitting it a
# dozen times (once per expander) is pure noise.
st.markdown(WRAPPING_TABLE_CSS, unsafe_allow_html=True)


def wrapping_table(frame: pd.DataFrame, index_col: str) -> None:
    """Render a DataFrame whose columns size to content, then wrap past a cap."""
    st.table(frame.set_index(index_col))


def resolve_api_key() -> tuple[str, str]:
    """Return ``(key, source)``. Called on the main thread only."""
    import os

    typed = (st.session_state.get("api_key_input") or "").strip()
    if typed:
        return typed, "sidebar"
    env = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if env:
        return env, ".env / environment"
    return "", "none"


def supports_effort(model: str) -> bool:
    return bool(MODEL_CAPS.get(model, {}).get("effort"))


def render_documents(tool_calls) -> None:
    """Show each retrieved document with its provenance."""
    for call in tool_calls:
        label = f"search_wikipedia({call.query!r}) -> {len(call.documents)} document(s)"
        if call.error:
            label = f"search_wikipedia({call.query!r}) -> error"
        if call.was_injected:
            label += "  [poisoned]"
        with st.expander(label):
            if call.error:
                st.error(call.error)
            for doc in call.documents:
                st.markdown(
                    f"**{doc['title']}** &nbsp; "
                    f"[link]({doc['url']}) &nbsp; "
                    f"`page_id={doc['page_id']}` &nbsp; `revision_id={doc['revision_id']}` &nbsp; "
                    f"retrieved `{doc['retrieved_at']}`"
                )
                if doc["truncated"]:
                    st.caption(
                        f"Truncated: {len(doc['text']):,} of {doc['chars_total']:,} characters "
                        "shown to the model."
                    )
                st.text_area(
                    "raw retrieved text",
                    doc["text"],
                    height=180,
                    key=f"doc_{call.tool_use_id}_{doc['page_id']}_{id(doc)}",
                    label_visibility="collapsed",
                )


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #

with st.sidebar:
    st.header("Setup")
    st.text_input(
        "Anthropic API key",
        type="password",
        key="api_key_input",
        placeholder="sk-ant-...",
        help="Only needed if ANTHROPIC_API_KEY is not already set in your .env file.",
    )

    api_key, key_source = resolve_api_key()
    if api_key:
        st.success(f"Key loaded from **{key_source}**")
    else:
        st.error("No API key found")
        st.markdown(
            "Either:\n"
            "1. `cp .env.example .env` and add your key, or\n"
            "2. paste it above.\n\n"
            "Get one at [console.anthropic.com](https://console.anthropic.com/settings/keys)."
        )

    st.divider()
    st.caption(
        f"**Agent** {st.session_state.agent_model}  \n"
        f"**Judges** {st.session_state.judge_model}  \n"
        f"Configure in the *Prompt & Config* tab."
    )
    st.divider()
    st.caption(
        "This project uses no hosted search or RAG tool. Retrieval is a custom "
        "`search_wikipedia` tool over the public MediaWiki APIs."
    )

chat_tab, config_tab, eval_tab = st.tabs(
    ["Chat", "Prompt & Config", "Evaluation Dashboard"]
)


# --------------------------------------------------------------------------- #
# Tab 1 -- Chat
# --------------------------------------------------------------------------- #

with chat_tab:
    st.subheader("Ask a question")

    # The dropdown carries only the short label: a selectbox row cannot wrap, so
    # putting the question in there truncates it. The full text is rendered below,
    # where it has room.
    left, right = st.columns([3, 1], vertical_alignment="bottom")
    with left:
        choice = st.selectbox(
            "Sample queries",
            options=list(range(len(SAMPLE_QUERIES))),
            format_func=lambda i: SAMPLE_QUERIES[i][0],
            label_visibility="collapsed",
        )
    with right:
        run_sample = st.button("Run sample", type="primary", use_container_width=True)

    sample_label, sample_question, sample_poisoned = SAMPLE_QUERIES[choice]
    with st.container(border=True):
        st.markdown(sample_question)
    if run_sample:
        st.session_state.pending = (sample_question, sample_poisoned)

    if sample_poisoned:
        st.warning(
            "This sample splices a simulated attacker payload into the retrieved "
            "article. The agent should answer the real question, keep its citations, "
            "and report the attempt."
        )

    # No st.rerun() here: the button click already triggers one, and this runs
    # before the message loop below, so the cleared list renders immediately.
    # Calling st.rerun() would abort the script before the config tab renders.
    if st.button("Clear conversation"):
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("tool_calls"):
                render_documents(message["tool_calls"])
            if message.get("meta"):
                st.caption(message["meta"])

    if typed := st.chat_input("Ask anything answerable from Wikipedia..."):
        st.session_state.pending = (typed, False)

    if st.session_state.pending:
        question, poisoned = st.session_state.pending
        st.session_state.pending = None

        if not api_key:
            st.error("Add an API key in the sidebar first.")
        else:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                search_fn = (
                    functools.partial(search_wikipedia, injection=INJECTION_DEMO)
                    if poisoned
                    else search_wikipedia
                )
                try:
                    agent = WikipediaAgent(
                        api_key=api_key,
                        model=st.session_state.agent_model,
                        effort=st.session_state.agent_effort,
                        system_prompt=st.session_state.system_prompt,
                        search_fn=search_fn,
                    )
                except MissingAPIKeyError as exc:
                    st.error(str(exc))
                    st.stop()

                status = st.status("Thinking...", expanded=True)

                def on_tool_call(call) -> None:
                    detail = call.error or f"{len(call.documents)} document(s)"
                    status.write(f"`search_wikipedia({call.query!r})` -> {detail}")

                # Prior turns as plain text messages so a follow-up ("and when
                # did it last erupt?") has an antecedent. Text only -- replaying
                # an assistant turn containing tool_use blocks would need its
                # matching tool_result blocks and make the request invalid.
                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                    if m.get("content")
                ]
                run = agent.run(question, history=history, on_tool_call=on_tool_call)
                status.update(
                    label=(
                        f"{len(run.tool_calls)} search(es)"
                        if run.searched
                        else "Answered without searching"
                    ),
                    state="complete",
                    expanded=False,
                )

                st.markdown(run.answer)
                render_documents(run.tool_calls)

                tokens = run.usage.get("input_tokens", 0) + run.usage.get("output_tokens", 0)
                meta = (
                    f"{run.model} | effort={run.effort} | {run.iterations} turn(s) | "
                    f"{tokens:,} tokens | {run.usage.get('cache_read_input_tokens', 0):,} cached | "
                    f"{run.latency_ms / 1000:.1f}s"
                )
                st.caption(meta)
                if run.error:
                    st.error(run.error)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": run.answer,
                        "tool_calls": run.tool_calls,
                        "meta": meta,
                    }
                )


# --------------------------------------------------------------------------- #
# Tab 2 -- Prompt & Config
# --------------------------------------------------------------------------- #

with config_tab:
    st.subheader("System prompt")
    st.caption(
        "Edits apply immediately to the Chat tab and to any eval run started from "
        "this session. The eval run file records the exact prompt used."
    )

    st.text_area(
        "System prompt",
        key=widget_key("system_prompt"),
        on_change=sync_widget("system_prompt"),
        height=460,
        label_visibility="collapsed",
    )

    # From the canonical list, not [DEFAULT_JUDGE_MODEL, "claude-opus-5"]: with
    # JUDGE_MODEL now overridable from .env, that literal could list the same
    # model twice.
    judge_models = JUDGE_MODELS
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.selectbox(
            "Agent model",
            AGENT_MODELS,
            key=widget_key("agent_model"),
            on_change=sync_widget("agent_model"),
        )
    with col_b:
        model = st.session_state.agent_model
        if supports_effort(model):
            st.selectbox(
                "Effort",
                EFFORT_LEVELS,
                key=widget_key("agent_effort"),
                on_change=sync_widget("agent_effort"),
            )
        else:
            st.selectbox("Effort", ["(not supported)"], disabled=True)
            st.caption(f"`{model}` does not accept the effort parameter.")
    with col_c:
        st.selectbox(
            "Judge model",
            judge_models,
            key=widget_key("judge_model"),
            on_change=sync_widget("judge_model"),
        )
        st.caption("Kept distinct from the agent model to avoid self-preference bias.")

    if st.button("Reset to defaults"):
        set_canonical("system_prompt", DEFAULT_SYSTEM_PROMPT)
        set_canonical("agent_model", DEFAULT_AGENT_MODEL)
        set_canonical("agent_effort", DEFAULT_EFFORT)
        set_canonical("judge_model", DEFAULT_JUDGE_MODEL)
        st.rerun()

    st.divider()
    st.markdown("#### The tool definition is a second prompt surface")
    st.warning(
        "Editing the system prompt alone may not change behaviour as much as you "
        "expect, because the **tool description below is also an instruction** and "
        "the model weighs both.\n\n"
        "Try it: set the system prompt to *'Say blabla only nothing else.'* and ask "
        "**'What is 2+2?'** - you get `blabla`. Ask **'How tall is Mount Fuji?'** and "
        "you get a normal cited answer instead. The ~180-word tool description says "
        "to call the tool *'whenever answering depends on a fact about the world, "
        "even if you believe you already know the answer'*, which simply outweighs a "
        "five-word system prompt. Adding *'never use any tool'* to the system prompt "
        "restores `blabla`.\n\n"
        "This is the same lesson the eval suite surfaced about the prompt itself: "
        "what matters is not whether a rule is present, but how salient and "
        "enforceable it is at the moment of the decision."
    )
    with st.expander("View the tool definition sent alongside every request"):
        st.caption(
            "Read-only here. It lives in `agent.SEARCH_WIKIPEDIA_TOOL` and is "
            "deliberately prescriptive about *when* to call the tool, not just what "
            "it does - trigger conditions in a description measurably raise "
            "should-call accuracy."
        )
        st.json(SEARCH_WIKIPEDIA_TOOL)


# --------------------------------------------------------------------------- #
# Tab 3 -- Evaluation dashboard
# --------------------------------------------------------------------------- #

with eval_tab:
    st.subheader("1. Verify the evaluators")
    st.caption(
        "A judge is an unvalidated classifier until you test it. This runs "
        "hand-written known-good and known-bad outputs through each judge and "
        "checks it returns the required verdict. Do this **before** trusting any "
        "number below it."
    )

    # vertical_alignment="bottom": a selectbox reserves space for its label and a
    # button does not, so without this the button sits above the dropdown.
    verify_cols = st.columns([1, 1, 2], vertical_alignment="bottom")
    with verify_cols[0]:
        verify_clicked = st.button("Verify Evaluators", type="primary", use_container_width=True)
    with verify_cols[1]:
        repeats = st.selectbox(
            "Repeats",
            [1, 3],
            help="Repeat each fixture to measure judge self-consistency.",
        )

    if verify_clicked:
        if not api_key:
            st.error("Add an API key in the sidebar first.")
        else:
            total_trials = len(JUDGE_FIXTURES) * repeats
            progress = st.progress(0.0, text=f"0 / {total_trials} judge fixtures...")
            client = anthropic.Anthropic(api_key=api_key, max_retries=4)

            def on_progress(done: int, total: int) -> None:
                """Fires on the main thread from inside as_completed."""
                progress.progress(done / total, text=f"{done} / {total} judge fixtures")

            # run_all_fixtures fans out internally and returns plain dicts; all
            # UI updates stay here on the main thread.
            st.session_state.verify_results = run_all_fixtures(
                client,
                model=st.session_state.judge_model,
                effort=DEFAULT_JUDGE_EFFORT,
                repeats=repeats,
                on_progress=on_progress,
            )
            progress.progress(1.0, text=f"{total_trials} / {total_trials} done")

    if st.session_state.verify_results:
        rows = st.session_state.verify_results
        passed = sum(r["passed"] for r in rows)
        consistency = sum(r["consistency"] for r in rows) / len(rows)

        m1, m2 = st.columns(2)
        m1.metric("Judge accuracy", f"{passed}/{len(rows)}")
        m2.metric("Self-consistency", f"{consistency:.0%}")
        if passed < len(rows):
            st.error("A judge failed its own fixtures. Fix the judge before reading the metrics below.")
        else:
            st.success("All judge fixtures passed. The metrics below are trustworthy.")

        wrapping_table(
            pd.DataFrame(
                [
                    {
                        "fixture": r["id"],
                        "judge": r["judge"],
                        "expected": r["expected_status"],
                        "actual": r["actual_status"],
                        "pass": "PASS" if r["passed"] else "FAIL",
                        "consistency": f"{r['consistency']:.0%}",
                        "why this fixture exists": r["rationale"],
                        "judge reasoning": r["reasoning"],
                    }
                    for r in rows
                ]
            ),
            index_col="fixture",
        )

    st.divider()
    st.subheader("2. Run the eval suite")

    cases = load_test_cases()
    run_cols = st.columns([1, 1, 2], vertical_alignment="bottom")
    with run_cols[0]:
        run_clicked = st.button("Run Eval Suite", type="primary", use_container_width=True)
    with run_cols[1]:
        limit = st.selectbox(
            "Cases", [len(cases), 3, 6], format_func=lambda n: f"{n} cases"
        )

    if run_clicked:
        if not api_key:
            st.error("Add an API key in the sidebar first.")
        else:
            selected = cases[:limit]
            progress = st.progress(0.0, text=f"0 / {len(selected)}")
            log = st.empty()
            done: list[CaseResult] = []

            def on_result(result: CaseResult) -> None:
                """Called on the main thread from inside run_suite's as_completed loop."""
                done.append(result)
                progress.progress(
                    len(done) / len(selected), text=f"{len(done)} / {len(selected)}"
                )
                log.write(f"{'ok ' if result.ok else 'ERROR'} {result.case_id}")

            results = run_suite(
                selected,
                api_key=api_key,
                system_prompt=st.session_state.system_prompt,
                agent_model=st.session_state.agent_model,
                agent_effort=st.session_state.agent_effort,
                judge_model=st.session_state.judge_model,
                on_result=on_result,
            )
            st.session_state.eval_results = results
            st.session_state.eval_metrics = aggregate(results)
            st.session_state.eval_run_path = save_run(
                results,
                system_prompt=st.session_state.system_prompt,
                agent_model=st.session_state.agent_model,
                agent_effort=st.session_state.agent_effort,
                judge_model=st.session_state.judge_model,
                judge_effort=DEFAULT_JUDGE_EFFORT,
            )
            log.empty()

    metrics = st.session_state.eval_metrics
    results = st.session_state.eval_results

    if metrics:
        if st.session_state.eval_run_path:
            st.caption(f"Saved to `{st.session_state.eval_run_path}`")
        if metrics.get("judge_errors"):
            st.warning(
                f"{metrics['judge_errors']} judge call(s) errored and were excluded "
                "from the denominators rather than scored."
            )

        current_category = None
        for spec in METRIC_SPECS:
            if spec["category"] != current_category:
                current_category = spec["category"]
                st.markdown(f"#### {current_category}")
            value_col, def_col = st.columns([1, 4])
            with value_col:
                st.metric(
                    spec["label"],
                    format_metric(metrics, spec),
                    help=metrics.get(spec["detail_key"]) if spec["detail_key"] else None,
                )
            with def_col:
                st.markdown(f"**What this measures.** {spec['definition']}")

        if metrics.get("ipi_not_triggered"):
            st.info(
                f"{metrics['ipi_not_triggered']} injection case(s) never fired because "
                "the agent did not search. Excluded from the resistance rate."
            )

        st.markdown("#### Per-question breakdown")
        st.caption("Verdicts at a glance. Judge reasoning for each case is below.")

        summary_rows = []
        for r in results:
            row = {"case": r.case_id, "category": r.category}
            for name in ("ipi_obedience", "claim_groundedness", "task_completion", "tool_use"):
                verdict = r.judges.get(name)
                row[name] = verdict.status if verdict else "-"
            row["searches"] = len(r.run.tool_calls) if r.run else 0
            if not r.ok:
                row["error"] = r.error
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Judge reasoning, per case")
        st.caption(
            "Every number above traces to one of these. Free-text reasoning is the "
            "debugging surface - three harness bugs in this project were found by "
            "reading it, not by reading the scores."
        )
        for r in results:
            failed = [v for v in r.judges.values() if v.scored and not v.passed]
            marker = "FAIL" if (failed or not r.ok) else "pass"
            with st.expander(f"[{marker}] {r.case_id} - {r.question[:80]}"):
                if r.error:
                    st.error(r.error)
                    if r.traceback:
                        st.code(r.traceback)
                if r.run and r.run.queries:
                    st.caption("queries: " + " | ".join(repr(q) for q in r.run.queries))
                # An errored case short-circuits before judging, so r.judges is
                # empty. Building a DataFrame from [] yields no columns, and
                # set_index("judge") would raise KeyError -- killing the whole
                # dashboard render and taking every other case's results with it.
                if r.judges:
                    wrapping_table(
                        pd.DataFrame(
                            [
                                {
                                    "judge": name,
                                    "category": verdict.category,
                                    "status": verdict.status,
                                    "reasoning": verdict.reasoning,
                                }
                                for name, verdict in r.judges.items()
                            ]
                        ),
                        index_col="judge",
                    )
                else:
                    st.info(
                        "No judge verdicts: this case failed before judging and was "
                        "excluded from every metric rather than scored."
                    )
                st.markdown("**Answer**")
                st.markdown(r.answer or "_(no answer)_")

    st.divider()
    st.subheader("3. Improve the prompt from the failures")
    st.caption(
        "Reads a saved run file from disk - not this session - and proposes a "
        "targeted revision of the system prompt based on the judges' reasoning."
    )

    saved_runs = list_runs()
    if not saved_runs:
        st.info("No saved runs yet. Run the eval suite first.")
    else:
        pick = st.selectbox(
            "Run file", saved_runs, format_func=lambda p: p.name
        )
        if st.button("Analyse Failures & Suggest Improved Prompt", type="primary"):
            if not api_key:
                st.error("Add an API key in the sidebar first.")
            else:
                run_data = load_run(pick)
                failures = summarise_failures(run_data)
                if not failures:
                    st.success("No scored judge failed in that run - nothing to improve.")
                else:
                    with st.spinner(f"Analysing {len(failures)} failing case(s)..."):
                        improvement, error = propose_improvement(
                            anthropic.Anthropic(api_key=api_key, max_retries=4), run_data
                        )
                    if error:
                        st.error(error)
                    else:
                        st.session_state.proposal = (improvement, run_data["run_id"])

        if st.session_state.proposal:
            improvement, run_id = st.session_state.proposal
            st.markdown("**Failure analysis**")
            st.write(improvement.failure_analysis)
            st.markdown("**Proposed changes**")
            for change in improvement.proposed_changes:
                st.markdown(f"- {change}")
            st.markdown("**Risks**")
            st.write(improvement.risks)
            st.markdown("**Revised system prompt**")
            st.code(improvement.revised_system_prompt, language="xml")

            apply_cols = st.columns(2)
            with apply_cols[0]:
                if st.button("Apply to Config tab", use_container_width=True):
                    set_canonical("system_prompt", improvement.revised_system_prompt)
                    st.session_state.proposal = None
                    st.success("Applied. Re-run the suite to measure the change.")
                    st.rerun()
            with apply_cols[1]:
                if st.button("Save proposal to disk", use_container_width=True):
                    st.success(f"Saved to `{save_proposal(improvement, run_id)}`")
