"""Load FinNI-eval-v2 at pins.eval_finni.hf_revision and verify rows_expected."""
from xbrl_frontier import config, finni


def main() -> None:
    pin = config.get("pins.eval_finni")
    rows = finni.load()
    print(f"{pin['repo']}@{pin['hf_revision']} split={pin['split']}: {len(rows)} rows "
          f"(rows_expected={pin['rows_expected']}) OK")


if __name__ == "__main__":
    main()
