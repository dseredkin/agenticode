#!/usr/bin/env python3
"""Create test issues for AgentiCode validation."""

import argparse
import os
import time

from github import Github

# Test issues for TheAlgorithms/Python repository
TEST_ISSUES = [
    {
        "title": "Add function to check if number is perfect square",
        "body": "Add a function `is_perfect_square(n: int) -> bool` in `maths/perfect_square.py` that returns True if n is a perfect square. Include doctests.",
    },
    {
        "title": "Add function to calculate factorial iteratively",
        "body": "Add a function `factorial_iterative(n: int) -> int` in `maths/factorial.py` that calculates factorial using a loop instead of recursion. Include doctests.",
    },
    {
        "title": "Add function to find GCD using Euclidean algorithm",
        "body": "Add a function `gcd_euclidean(a: int, b: int) -> int` in `maths/gcd.py` that finds the greatest common divisor. Include doctests.",
    },
    {
        "title": "Add function to check if string is palindrome",
        "body": "Add a function `is_palindrome(s: str) -> bool` in `strings/palindrome.py` that checks if a string is a palindrome (ignoring case and spaces). Include doctests.",
    },
    {
        "title": "Add function to reverse a string",
        "body": "Add a function `reverse_string(s: str) -> str` in `strings/reverse.py` that reverses a string. Include doctests.",
    },
    {
        "title": "Add function to count vowels in string",
        "body": "Add a function `count_vowels(s: str) -> int` in `strings/count_vowels.py` that counts vowels in a string. Include doctests.",
    },
    {
        "title": "Add function to check if number is prime",
        "body": "Add a function `is_prime_simple(n: int) -> bool` in `maths/prime_simple.py` that checks if n is prime using trial division. Include doctests.",
    },
    {
        "title": "Add function to calculate sum of digits",
        "body": "Add a function `sum_of_digits(n: int) -> int` in `maths/sum_of_digits.py` that returns the sum of digits of a number. Include doctests.",
    },
    {
        "title": "Add function to find maximum in list",
        "body": "Add a function `find_max(arr: list) -> int` in `searches/find_max.py` that finds the maximum element without using built-in max(). Include doctests.",
    },
    {
        "title": "Add function to find minimum in list",
        "body": "Add a function `find_min(arr: list) -> int` in `searches/find_min.py` that finds the minimum element without using built-in min(). Include doctests.",
    },
    {
        "title": "Add function to calculate power recursively",
        "body": "Add a function `power_recursive(base: int, exp: int) -> int` in `maths/power.py` that calculates base^exp using recursion. Include doctests.",
    },
    {
        "title": "Add function to check if year is leap year",
        "body": "Add a function `is_leap_year(year: int) -> bool` in `maths/leap_year.py` that checks if a year is a leap year. Include doctests.",
    },
    {
        "title": "Add function to convert Celsius to Fahrenheit",
        "body": "Add a function `celsius_to_fahrenheit(c: float) -> float` in `conversions/temperature.py` that converts Celsius to Fahrenheit. Include doctests.",
    },
    {
        "title": "Add function to calculate average of list",
        "body": "Add a function `calculate_average(numbers: list) -> float` in `maths/average.py` that calculates the average of a list of numbers. Include doctests.",
    },
    {
        "title": "Add function to count words in string",
        "body": "Add a function `count_words(s: str) -> int` in `strings/word_count.py` that counts words in a string. Include doctests.",
    },
    {
        "title": "Add function to find second largest element",
        "body": "Add a function `second_largest(arr: list) -> int` in `searches/second_largest.py` that finds the second largest element in a list. Include doctests.",
    },
    {
        "title": "Add function to check Armstrong number",
        "body": "Add a function `is_armstrong(n: int) -> bool` in `maths/armstrong.py` that checks if n is an Armstrong number. Include doctests.",
    },
    {
        "title": "Add function to generate Fibonacci sequence",
        "body": "Add a function `fibonacci_list(n: int) -> list` in `maths/fibonacci_list.py` that returns first n Fibonacci numbers as a list. Include doctests.",
    },
    {
        "title": "Add function to remove duplicates from list",
        "body": "Add a function `remove_duplicates(arr: list) -> list` in `other/remove_duplicates.py` that removes duplicates while preserving order. Include doctests.",
    },
    {
        "title": "Add function to check if two strings are anagrams",
        "body": "Add a function `are_anagrams(s1: str, s2: str) -> bool` in `strings/anagram.py` that checks if two strings are anagrams. Include doctests.",
    },
]


def create_issues(token: str, repo_name: str, dry_run: bool = False) -> list[int]:
    """Create test issues in the repository."""
    gh = Github(token)
    repo = gh.get_repo(repo_name)

    created_issues = []

    for i, issue_data in enumerate(TEST_ISSUES, 1):
        title = f"[validation] {issue_data['title']}"
        body = f"{issue_data['body']}\n\n---\n_Validation issue {i}/20_"

        if dry_run:
            print(f"[DRY RUN] Would create: {title}")
        else:
            issue = repo.create_issue(title=title, body=body)
            created_issues.append(issue.number)
            print(f"Created issue #{issue.number}: {title}")
            time.sleep(1)  # Rate limiting

    return created_issues


def main():
    parser = argparse.ArgumentParser(description="Create test issues for validation")
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
        "--dry-run",
        action="store_true",
        help="Print issues without creating them",
    )
    args = parser.parse_args()

    if not args.token:
        print("Error: GitHub token required (--token or GITHUB_TOKEN env var)")
        return 1

    if not args.repo:
        print("Error: Repository required (--repo or GITHUB_REPOSITORY env var)")
        return 1

    print(f"Creating {len(TEST_ISSUES)} validation issues in {args.repo}...")
    issues = create_issues(args.token, args.repo, args.dry_run)

    if not args.dry_run:
        print(f"\nCreated {len(issues)} issues: {issues}")
        print("Run validation_metrics.py later to check approval rate.")

    return 0


if __name__ == "__main__":
    exit(main())
