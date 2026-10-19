import pytest

from app.agent.catalogue import (
    FACT_VERSION,
    BrandFactCatalogue,
    CatalogueError,
    UnknownBrandError,
    UnknownToolError,
    execute_tool,
    get_brand_facts,
)


def test_catalogue_loads_known_brands() -> None:
    catalogue = BrandFactCatalogue()

    assert catalogue.brand_ids == (
        "cleartrail",
        "cloudnote",
        "stonewave",
    )


def test_lookup_returns_facts_for_requested_brand() -> None:
    result = get_brand_facts("cleartrail")

    assert result["fact_version"] == FACT_VERSION
    assert result["brand_id"] == "cleartrail"
    assert len(result["facts"]) > 0

    assert all(
        fact["brand_id"] == "cleartrail"
        for fact in result["facts"]
    )


def test_lookup_preserves_catalogue_fields() -> None:
    result = get_brand_facts("cleartrail")
    fact = result["facts"][0]

    assert set(fact.keys()) == {
        "fact_id",
        "brand_id",
        "model_id",
        "field",
        "value",
        "value_type",
        "unit",
        "condition",
        "valid_from",
        "valid_to",
    }


def test_unknown_brand_raises_error() -> None:
    with pytest.raises(UnknownBrandError):
        get_brand_facts("unknown_brand")


@pytest.mark.parametrize("brand_id", ["", None, 123])
def test_invalid_brand_argument_raises_error(
    brand_id: object,
) -> None:
    with pytest.raises(CatalogueError):
        get_brand_facts(brand_id)  # type: ignore[arg-type]


def test_executor_runs_allowed_tool() -> None:
    result = execute_tool(
        "get_brand_facts",
        {"brand_id": "cloudnote"},
    )

    assert result["brand_id"] == "cloudnote"
    assert result["fact_version"] == FACT_VERSION
    assert result["facts"]


def test_executor_rejects_unknown_tool() -> None:
    with pytest.raises(UnknownToolError):
        execute_tool(
            "execute_arbitrary_code",
            {"brand_id": "cleartrail"},
        )


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"brand_id": "cleartrail", "extra": "not-allowed"},
        {"brand_id": ""},
        {"brand_id": 123},
    ],
)
def test_executor_rejects_invalid_arguments(
    arguments: dict[str, object],
) -> None:
    with pytest.raises(CatalogueError):
        execute_tool("get_brand_facts", arguments)


def test_executor_rejects_unknown_requested_brand() -> None:
    with pytest.raises(UnknownBrandError):
        execute_tool(
            "get_brand_facts",
            {"brand_id": "made_up_brand"},
        )