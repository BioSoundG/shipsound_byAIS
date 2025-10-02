#!/usr/bin/env python
"""
Test runner script for shipsound_byAIS project
"""
import unittest
import os
import sys

# プロジェクトのルートディレクトリをパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run_tests():
    """実装した機能のテストを実行"""
    # test ディレクトリからテストをロード
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(os.path.dirname(__file__), pattern="test_*.py")

    # テスト実行
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)

    # テスト結果に基づいて終了コードを設定
    sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    run_tests()
