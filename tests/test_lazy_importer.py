import pytest
from types import ModuleType
from myllm.utils.lazy import LazyImporter


def test_lazy_import_is_module():
    """Tests that the lazy importer is a module."""
    math = LazyImporter("math")
    assert isinstance(math, ModuleType)


def test_lazy_import_success():
    """Tests that a module is imported when an attribute is accessed."""
    math = LazyImporter("math")
    # The module should not be loaded yet
    assert math._module is None
    # Access an attribute to trigger the import
    assert math.sqrt(4) == 2.0
    # The module should now be loaded
    assert math._module is not None
    assert math.__name__ == "math"


def test_lazy_import_failure():
    """Tests that an ImportError is raised for a non-existent module."""
    non_existent_module = LazyImporter("non_existent_module_for_testing_purposes")
    with pytest.raises(ImportError):
        _ = non_existent_module.some_attribute


def test_lazy_importer_dir():
    """Tests that dir() on the lazy importer returns the real module's attributes."""
    math = LazyImporter("math")
    import math as real_math

    # Access an attribute to trigger the import
    _ = math.sqrt

    # Now dir() should work
    lazy_dir = dir(math)
    real_dir = dir(real_math)
    assert set(lazy_dir) == set(real_dir)

def test_lazy_importer_special_methods():
    """Tests that special methods are forwarded correctly."""
    math = LazyImporter("math")
    assert math.__name__ == "math" 