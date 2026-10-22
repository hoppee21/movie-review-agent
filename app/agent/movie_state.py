"""Pure conversion between LLM movie analysis and executable movie state."""

import json
import re
import unicodedata

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from app.agent.schemas import (
    Action,
    AgentState,
    ClarificationIssue,
    ClarificationRecord,
    MovieAnalysis,
    MovieConstraint,
    MovieMention,
    MovieTarget,
    RunResult,
    RunStatus,
)


def movie_updates(state: AgentState, analysis: MovieAnalysis) -> AgentState:
    """Apply the recognition contract; only external facts may establish identity."""
    texts = [state.get("user_query", "")]
    for raw in state.get("clarification_history", []):
        record = ClarificationRecord.model_validate(raw)
        texts.extend([record.text or "", record.selected_label or ""])
    mentions = [item for item in analysis.mentions if item.role == "target"]
    targets: list[MovieTarget] = []
    for item in mentions:
        if re.fullmatch(r"[这那该此](?:一)?(?:部|个)?(?:电影|影片|片子|片)?", item.query.strip()):
            targets.extend(
                MovieTarget.model_validate(target)
                for target in state.get("movie_targets", [])
            )
        else:
            targets.append(_mention_target(item, len(targets) + 1, texts))
    if analysis.kind == "general_movie" and targets:
        analysis = analysis.model_copy(update={"kind": "specific_movie"})
    updates: AgentState = {
        "movie_analysis": analysis,
        "movie_targets": targets,
        "pending_issues": [],
    }
    reason = {
        "non_movie": "暂时无法处理非电影请求。",
        "general_movie": "暂时无法处理未指向具体作品的电影泛请求。",
    }.get(analysis.kind)
    if not reason and (
        len(targets) > 1 or any(item.scope == "series" for item in mentions)
    ):
        reason = "当前仅支持单部电影，暂不支持多部作品或整个系列的联合处理。"
    if reason:
        updates["final_result"] = RunResult(
            status=RunStatus.UNSUPPORTED,
            reason=reason,
            data={"kind": analysis.kind, "movie_queries": [item.query for item in mentions]},
        )
    elif not targets:
        updates["pending_issues"] = [
            ClarificationIssue(
                issue_id=_analysis_issue_id(state),
                kind="missing_movie",
                question=(analysis.clarification_question or "").strip()
                or "你指哪部电影？请补充片名、年份或其他线索。",
            )
        ]
    else:
        updates["pending_issues"] = [
            ClarificationIssue(
                issue_id=_analysis_issue_id(state),
                kind="movie_lookup",
                target_id=targets[0].target_id,
                question="核实电影身份与版本。",
                suggested_actions=[Action.RESOLVE_MOVIE],
            )
        ]
    return updates


def _mention_target(mention: MovieMention, index: int, texts: list[str]) -> MovieTarget:
    # A guessed identifier must not become an exact lookup bypassing confirmation.
    pattern = re.escape(mention.query)
    if is_identifier(mention.query):
        pattern = rf"(?<![A-Za-z0-9]){pattern}(?![A-Za-z0-9])"
    query = mention.query if any(re.search(pattern, text) for text in texts) else texts[0]
    return MovieTarget(
        target_id=f"movie-{index}",
        query=query or mention.query,
        release_year=_explicit_number(mention.release_year, texts, 1888, 2100),
        installment=_explicit_number(mention.installment, texts, 1, 100),
        search_hints=[hint for hint in mention.search_hints if not is_identifier(hint)],
    )


def _explicit_number(
    clue: MovieConstraint | None, texts: list[str], minimum: int, maximum: int
) -> int | None:
    if clue is None or not minimum <= clue.value <= maximum:
        return None
    if clue.value not in numbers_in_text(clue.quote):
        return None
    negative = r"不是|不要|不看|排除|除了|非"
    if re.search(negative, clue.quote):
        return None
    # A quote cannot cut a number in half or read "I" from "IMDb".
    number_chars = r"\d零〇一二两三四五六七八九十首"
    pattern = re.escape(clue.quote)
    if re.fullmatch(rf"[{number_chars}]", clue.quote[0]):
        pattern = rf"(?<![{number_chars}]){pattern}"
    if re.fullmatch(rf"[{number_chars}]", clue.quote[-1]):
        pattern = rf"{pattern}(?![{number_chars}])"
    if clue.quote[0] in "IVXivx":
        pattern = rf"(?<![A-Za-z0-9_]){pattern}"
    if clue.quote[-1] in "IVXivx":
        pattern = rf"{pattern}(?![A-Za-z0-9_])"
    for text in texts:
        for match in re.finditer(pattern, text):
            prefix = text[max(0, match.start() - 6):match.start()]
            if not re.search(rf"(?:{negative})\s*$", prefix):
                return clue.value
    return None


def numbers_in_text(text: str) -> set[int]:
    """Read literal Arabic/Chinese/Roman numbers, without inferring movie facts."""
    text = unicodedata.normalize("NFKC", text)
    result = {int(value) for value in re.findall(r"\d+", text)}
    digits = {char: index for index, char in enumerate("零一二三四五六七八九")}
    digits.update({"〇": 0, "两": 2, "首": 1})
    for value in re.findall(r"[零〇一二两三四五六七八九十首]+", text):
        if "十" in value:
            left, right = value.split("十", 1)
            result.add((digits.get(left, 1) if left else 1) * 10 + digits.get(right, 0))
        else:
            result.add(int("".join(str(digits[char]) for char in value)))
    roman = dict(
        zip(("I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"), range(1, 11))
    )
    result.update(
        roman[value.upper()]
        for value in re.findall(r"\b[IVX]+\b", text, re.I)
        if value.upper() in roman
    )
    return result


def movie_context(targets: list[MovieTarget]) -> str:
    if not targets:
        raise RuntimeError("Movie context requires at least one target")
    normalized = [MovieTarget.model_validate(target) for target in targets]
    return "\n".join(
        f"- {target_label(target)}"
        + (f" [{target.wikidata_id}]" if target.wikidata_id else "")
        for target in normalized
    )


def target_label(target: MovieTarget) -> str:
    title = target.english_title or target.query
    return f"{title} ({target.release_year})" if target.release_year else title


def clarification_messages(
    records: list[ClarificationRecord],
) -> list[BaseMessage]:
    messages: list[BaseMessage] = []
    for raw_record in records:
        record = ClarificationRecord.model_validate(raw_record)
        messages.append(
            AIMessage(
                content=json.dumps(
                    {
                        "issue_id": record.issue_id,
                        "question": record.question,
                        "options": record.options,
                    },
                    ensure_ascii=False,
                )
            )
        )
        answer = record.selected_label or ""
        if record.text:
            answer = f"{answer}；{record.text}" if answer else record.text
        messages.append(HumanMessage(content=answer))
    return messages


def is_identifier(value: str) -> bool:
    return bool(
        re.fullmatch(
            r"(?:tt\d{7,10}|Q\d+|douban\s*[:#]\s*\d+"
            r"|https?://(?:www\.)?(?:imdb\.com/title/tt\d{7,10}"
            r"|movie\.douban\.com/subject/\d+|wikidata\.org/(?:wiki|entity)/Q\d+)"
            r"(?:[/?#]\S*)?)",
            value.strip(),
            re.IGNORECASE,
        )
    )


def _analysis_issue_id(state: AgentState) -> str:
    return f"movie-analysis-{len(state.get('clarification_history', [])) + 1}"
