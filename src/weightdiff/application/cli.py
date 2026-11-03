import argparse
import sys
from pathlib import Path

from .collect import build_report
from .report import DISCLAIMER, flags_markdown, summary_markdown, to_json, to_markdown


def main(argv=None):
    parser = argparse.ArgumentParser(description='Compare model structure and small byte samples without loading weights.')
    parser.add_argument('A')
    parser.add_argument('B')
    parser.add_argument('--rev-a')
    parser.add_argument('--rev-b')
    parser.add_argument('--no-sample', action='store_true')
    parser.add_argument('--max-tensors', type=int, default=160)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--md', type=Path)
    args = parser.parse_args(argv)
    if args.max_tensors < 0:
        parser.error('--max-tensors must be nonnegative')
    try:
        report = build_report(args.A, args.B, args.rev_a, args.rev_b, not args.no_sample, args.max_tensors)
        for path, content in [(args.json, to_json(report)), (args.md, to_markdown(report))]:
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding='utf-8')
    except Exception as error:
        print(f'weightdiff: {error}', file=sys.stderr)
        return 1
    print(DISCLAIMER + '\n\n' + summary_markdown(report) + '\n\n' + flags_markdown(report))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
