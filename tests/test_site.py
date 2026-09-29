import pytest

from harmonybench.dataset import Item
from harmonybench.site import _check_public


def item(source="JANE DOE") -> Item:
    return Item(
        index=0,
        item_id="wrong_note-001",
        subset="wrong_note",
        image="images/x.png",
        image_sha256="",
        label={"measure": 3},
        meta={"source": source},
    )


def test_public_page_guard_accepts_clean_pages():
    _check_public("<html><body><h1>Ranking</h1><table></table></body></html>", [item()])


@pytest.mark.parametrize(
    "page,leak",
    [
        ('<img src="data:image/png;base64,AAAA">', "image"),
        ('<div class="said"><p>Bar 3: the alto G natural</p></div>', "responses"),
        ('<div class="answers"><details><summary>Opus</summary></details></div>', "responses"),
        ("<p>JANE DOE</p>", "arrangement"),
    ],
)
def test_public_page_guard_rejects_private_material(page, leak):
    with pytest.raises(ValueError, match=leak):
        _check_public(page, [item()])
