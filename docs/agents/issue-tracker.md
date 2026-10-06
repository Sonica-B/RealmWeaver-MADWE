# Issue tracker: GitHub Issues (Sonica-B/RealmWeaver-MADWE)

All planning artifacts for the skills in this repo (wayfinder maps, to-spec specs, code-review spec lookup) live in GitHub Issues, driven with the `gh` CLI (already authenticated on the owner's machine).

## Conventions
- Spec issues: title `Spec: <feature>`, body = the to-spec template, label `ready-for-agent` when agents may pick it up.
- Commit messages reference issues as `#123`; PR bodies say `Closes #123`.
- Fetch an issue: `gh issue view 123 --json title,body,labels,comments`.

## Wayfinding operations
- **Map**: one issue labelled `wayfinder:map`; body holds Destination / Notes / Decisions so far / Not yet specified / Out of scope. Tickets are **sub-issues** of the map (GitHub sub-issues API).
- **Ticket**: a sub-issue labelled `wayfinder:research` | `wayfinder:prototype` | `wayfinder:grilling` | `wayfinder:task`; body `## Question`. Create: `gh issue create --title ... --label ... --body ...`; attach as sub-issue: `gh api -X POST repos/Sonica-B/RealmWeaver-MADWE/issues/<map>/sub_issues -F sub_issue_id=<ticket_node_db_id>`.
- **Blocking**: GitHub issue dependencies (native, rendered in the UI): `gh api -X POST repos/Sonica-B/RealmWeaver-MADWE/issues/<ticket>/dependencies/blocked_by -F issue_id=<blocker_db_id>`.
- **Claim**: assign yourself: `gh issue edit <n> --add-assignee @me`. An open, unassigned, unblocked sub-issue is the frontier.
- **Frontier query** (gh's `--label` is AND, so filter in jq): `gh issue list --state open --limit 100 --json number,title,labels,assignees --jq '.[] | select((.assignees | length) == 0) | select([.labels[].name] | any(startswith("wayfinder:") and . != "wayfinder:map")) | "\(.number)	\(.title)"'` then drop any issue whose `gh api repos/.../issues/<n>/dependencies/blocked_by` list has open items.
- **Resolve**: post the answer as a comment (`gh issue comment <n> --body ...`), close (`gh issue close <n>`), append one line to the map's "Decisions so far" (`gh issue edit <map> --body-file ...`).
- Epics: label `epic`; one epic per subsystem; tickets reference their epic in the body.
