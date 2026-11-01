"""Turn JUnit XML test reports into a Markdown table for the GitHub Actions job summary.

Usage: python .github/scripts/junit_summary.py "<title>" report1.xml [report2.xml ...]
"""

import sys
import xml.etree.ElementTree as ET


def summarize(paths):
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "time": 0.0}
    failed = []
    for path in paths:
        root = ET.parse(path).getroot()
        suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
        for suite in suites:
            for key in ("tests", "failures", "errors", "skipped"):
                totals[key] += int(suite.get(key, 0))
            totals["time"] += float(suite.get("time", 0))
            for case in suite.iter("testcase"):
                if case.find("failure") is not None or case.find("error") is not None:
                    failed.append(f"{case.get('classname')}::{case.get('name')}")
    totals["passed"] = totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"]
    return totals, failed


def to_markdown(title, totals, failed):
    status = "✅ all passed" if not failed else f"❌ {len(failed)} failed"
    lines = [
        f"### {title} — {status}",
        "",
        "| Tests | Passed | Failed | Errors | Skipped | Time (s) |",
        "|---|---|---|---|---|---|",
        f"| {totals['tests']} | {totals['passed']} | {totals['failures']} | {totals['errors']} "
        f"| {totals['skipped']} | {totals['time']:.2f} |",
        "",
    ]
    if failed:
        lines.append("**Failed tests:**")
        lines.extend(f"- `{name}`" for name in failed)
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: junit_summary.py <title> <report.xml> [...]")
    totals, failed = summarize(sys.argv[2:])
    print(to_markdown(sys.argv[1], totals, failed))
