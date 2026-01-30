#!/usr/bin/env python3
"""Calculate approval rate for AgentiCode PRs."""

import argparse
import os

from github import Github


def calculate_approval_rate(token: str, repo_name: str) -> dict:
    """Calculate ratio of approved PRs to open PRs."""
    gh = Github(token)
    repo = gh.get_repo(repo_name)

    # Get PRs created by agenticode (with feat: prefix)
    open_prs = list(repo.get_pulls(state="open"))
    closed_prs = list(repo.get_pulls(state="closed"))

    # Filter PRs with feat: prefix (created by Code Agent)
    agent_open = [pr for pr in open_prs if pr.title.startswith("feat:")]
    agent_closed = [pr for pr in closed_prs if pr.title.startswith("feat:")]

    # Count approved/merged
    approved = 0
    for pr in agent_closed:
        if pr.merged:
            approved += 1
        else:
            # Check if it was approved before closing
            reviews = pr.get_reviews()
            if any(r.state == "APPROVED" for r in reviews):
                approved += 1

    total = len(agent_open) + len(agent_closed)
    open_count = len(agent_open)

    return {
        "total_agent_prs": total,
        "open": open_count,
        "approved_or_merged": approved,
        "approval_rate": approved / total if total else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Calculate AgentiCode approval rate")
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="Repository in owner/repo format (or set GITHUB_REPOSITORY env var)",
    )
    args = parser.parse_args()

    if not args.token:
        print("Error: GitHub token required (--token or GITHUB_TOKEN env var)")
        return 1

    if not args.repo:
        print("Error: Repository required (--repo or GITHUB_REPOSITORY env var)")
        return 1

    result = calculate_approval_rate(args.token, args.repo)

    print(f"Repository: {args.repo}")
    print(f"Total Agent PRs: {result['total_agent_prs']}")
    print(f"Open: {result['open']}")
    print(f"Approved/Merged: {result['approved_or_merged']}")
    print(f"Approval Rate: {result['approval_rate']:.1%}")

    return 0


if __name__ == "__main__":
    exit(main())
