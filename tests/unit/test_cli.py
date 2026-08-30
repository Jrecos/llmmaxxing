from llmmaxxing.cli.main import build_parser


def test_cli_has_two_daemons():
    parser = build_parser()
    assert parser.parse_args(["gateway"]).command == "gateway"
    assert parser.parse_args(["control"]).command == "control"
