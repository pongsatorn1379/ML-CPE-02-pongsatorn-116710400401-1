"""รันงานทั้งหมด: ฝึกโมเดลก่อน แล้วแสดงตัวอย่างการทำนาย."""

from main import main
from test_cnn import test_cnn


if __name__ == "__main__":
    print("=== 1. Train and evaluate CNN ===")
    main()

    print("\n=== 2. Show prediction examples ===")
    test_cnn()

    print("\nFinished. See results in outputs/")
