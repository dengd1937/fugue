"""tests/unit/test_optional.py — _optional.require() 共享 helper 测试。"""

import pathlib

import pytest

from fugue._optional import require


class TestRequireSuccess:
    """require() 在模块存在时正常返回模块对象的测试。"""

    def test_returns_module_object_for_existing_module(self) -> None:
        """require("pathlib", ...) 应返回 pathlib 模块本身。"""
        mod = require("pathlib", extra="x")
        assert mod is pathlib

    def test_returned_module_has_expected_attribute(self) -> None:
        """返回的模块对象应具有 .Path 属性。"""
        mod = require("pathlib", extra="x")
        assert hasattr(mod, "Path")


class TestRequireFailure:
    """require() 在模块缺失时抛出 ImportError 的测试。"""

    def test_raises_import_error_for_absent_package(self) -> None:
        """不存在的包应抛 ImportError。"""
        with pytest.raises(ImportError):
            require("definitely_absent_pkg_xyz", extra="pdf")

    def test_error_message_contains_install_hint(self) -> None:
        """错误消息应含 pip install 'fugue[pdf]' 子串。"""
        with pytest.raises(ImportError) as excinfo:
            require("definitely_absent_pkg_xyz", extra="pdf")
        assert "pip install 'fugue[pdf]'" in str(excinfo.value)

    def test_exception_cause_is_not_none(self) -> None:
        """__cause__ 应保留原始异常链（raise ... from e）。"""
        with pytest.raises(ImportError) as excinfo:
            require("definitely_absent_pkg_xyz", extra="pdf")
        assert excinfo.value.__cause__ is not None

    def test_error_message_contains_module_name(self) -> None:
        """错误消息应含实际模块名。"""
        with pytest.raises(ImportError) as excinfo:
            require("definitely_absent_pkg_xyz", extra="pdf")
        assert "definitely_absent_pkg_xyz" in str(excinfo.value)


@pytest.mark.parametrize(
    "extra,expected_substring",
    [
        ("bge", "fugue[bge]"),
        ("pdf", "fugue[pdf]"),
        ("custom_extra", "fugue[custom_extra]"),
    ],
)
def test_extra_name_appears_in_error_message(extra: str, expected_substring: str) -> None:
    """extra 参数值应出现在错误消息中。"""
    with pytest.raises(ImportError) as excinfo:
        require("definitely_absent_pkg_xyz", extra=extra)
    assert expected_substring in str(excinfo.value)


def test_exact_error_message_format() -> None:
    """错误消息格式必须精确匹配规格（含中文标点与单引号）。"""
    module_name = "definitely_absent_pkg_xyz"
    extra = "pdf"
    expected_msg = f"可选依赖 '{module_name}' 未安装。请运行: pip install 'fugue[{extra}]'"
    with pytest.raises(ImportError) as excinfo:
        require(module_name, extra=extra)
    assert str(excinfo.value) == expected_msg
