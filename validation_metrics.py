#!/usr/bin/env python3
"""Calculate approval rate for validation issues."""

import argparse
import os

from github import Github


def calculate_validation_metrics(token: str, repo_name: str) -> dict:
    """Calculate approval rate for validation issues."""
    gh = Github(token)
    repo = gh.get_repo(repo_name)

    # Find validation issues
    issues = repo.get_issues(state="all")
    validation_issues = [i for i in issues if i.title.startswith("[validation]")]

    results = {
        "total": len(validation_issues),
        "with_pr": 0,
        "pr_open": 0,
        "pr_merged": 0,
        "pr_approved": 0,
        "pr_changes_requested": 0,
        "no_pr": 0,
        "details": [],
    }

    for issue in validation_issues:
        issue_data = {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "pr": None,
            "pr_state": None,
            "review_state": None,
        }

        # Find linked PR (search by issue number in title)
        prs = repo.get_pulls(state="all")
        linked_pr = None
        for pr in prs:
            if f"#{issue.number}" in pr.title or f"#{issue.number}" in (pr.body or ""):
                linked_pr = pr
                break

        if linked_pr:
            results["with_pr"] += 1
            issue_data["pr"] = linked_pr.number
            issue_data["pr_state"] = "merged" if linked_pr.merged else linked_pr.state

            if linked_pr.merged:
                results["pr_merged"] += 1
                issue_data["review_state"] = "merged"
            elif linked_pr.state == "open":
                results["pr_open"] += 1
                # Check reviews
                reviews = list(linked_pr.get_reviews())
                if reviews:
                    latest_review = reviews[-1]
                    issue_data["review_state"] = latest_review.state
                    if latest_review.state == "APPROVED":
                        results["pr_approved"] += 1
                    elif latest_review.state == "CHANGES_REQUESTED":
                        results["pr_changes_requested"] += 1
            else:
                # Closed without merge
                reviews = list(linked_pr.get_reviews())
                if reviews:
                    issue_data["review_state"] = reviews[-1].state
        else:
            results["no_pr"] += 1

        results["details"].append(issue_data)

    # Calculate approval rate
    successful = results["pr_merged"] + results["pr_approved"]
    results["approval_rate"] = successful / results["total"] if results["total"] else 0

    return results


def main():
    parser = argparse.ArgumentParser(description="Calculate validation approval rate")
    parser.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN"),
        help="GitHub token (or set GITHUB_TOKEN env var)",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="Repository in owner/repo format",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show details for each issue",
    )
    args = parser.parse_args()

    if not args.token:
        print("Error: GitHub token required (--token or GITHUB_TOKEN env var)")
        return 1

    if not args.repo:
        print("Error: Repository required (--repo or GITHUB_REPOSITORY env var)")
        return 1

    print(f"Calculating metrics for {args.repo}...\n")
    results = calculate_validation_metrics(args.token, args.repo)

    print("=" * 50)
    print("VALIDATION METRICS")
    print("=" * 50)
    print(f"Total validation issues: {results['total']}")
    print(f"With PR created:         {results['with_pr']}")
    print(f"  - Merged:              {results['pr_merged']}")
    print(f"  - Open (approved):     {results['pr_approved']}")
    print(f"  - Open (changes req):  {results['pr_changes_requested']}")
    print(f"  - Open (pending):      {results['pr_open'] - results['pr_approved'] - results['pr_changes_requested']}")
    print(f"Without PR:              {results['no_pr']}")
    print("=" * 50)
    print(f"APPROVAL RATE: {results['approval_rate']:.1%}")
    print("=" * 50)

    if args.verbose:
        print("\nDETAILS:")
        for d in results["details"]:
            pr_info = f"PR #{d['pr']} ({d['pr_state']}, {d['review_state']})" if d["pr"] else "no PR"
            print(f"  #{d['number']}: {pr_info}")

    return 0


if __name__ == "__main__":
    exit(main())
