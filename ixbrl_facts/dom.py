"""Minimal stdlib HTML tree with the BeautifulSoup calls FinTagging's serializer uses.

FinTagging builds its contexts with `BeautifulSoup(content, "lxml")` (libxml2's HTML parser). This module
reproduces the parts of that tree and API the serializer touches, on `html.parser`, so no third-party
parser is needed:
- tag and attribute names lowercased (`ix:nonfraction`), character references decoded;
- comments dropped; script/style text excluded from text extraction (bs4 >= 4.10 behaviour);
- void elements never take children; an end tag closes up to its matching open element, a stray end tag
  is ignored;
- libxml2's HTML auto-close: a block-level start tag closes an open `p` that is the current element.
"""
from html.parser import HTMLParser

VOID = frozenset({"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param",
                  "source", "track", "wbr"})
RAW_TEXT = frozenset({"script", "style"})
CLOSES_P = frozenset({"address", "article", "aside", "blockquote", "center", "dir", "div", "dl", "fieldset",
                      "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "menu", "nav",
                      "ol", "p", "pre", "section", "table", "ul"})


class Tag:
    __slots__ = ("name", "attrs", "children", "parent")

    def __init__(self, name: str, attrs: dict, parent: "Tag | None"):
        self.name = name
        self.attrs = attrs
        self.children: list = []
        self.parent = parent

    def get(self, key: str, default=None):
        return self.attrs.get(key, default)

    def _strings(self):
        stack = [iter(self.children)]
        while stack:
            for c in stack[-1]:
                if isinstance(c, str):
                    yield c
                elif c.name not in RAW_TEXT:
                    stack.append(iter(c.children))
                    break
            else:
                stack.pop()

    @property
    def text(self) -> str:
        return "".join(self._strings())

    def get_text(self, separator: str = "", strip: bool = False) -> str:
        if strip:
            return separator.join(s for s in (x.strip() for x in self._strings()) if s)
        return separator.join(self._strings())

    def _descendants(self):
        stack = [iter(self.children)]
        while stack:
            for c in stack[-1]:
                if isinstance(c, Tag):
                    yield c
                    stack.append(iter(c.children))
                    break
            else:
                stack.pop()

    def find_all(self, names, recursive: bool = True) -> list["Tag"]:
        names = {names} if isinstance(names, str) else set(names)
        pool = self._descendants() if recursive else (c for c in self.children if isinstance(c, Tag))
        return [t for t in pool if t.name in names]

    def find(self, name: str) -> "Tag | None":
        return next((t for t in self._descendants() if t.name == name), None)

    def find_parent(self, names) -> "Tag | None":
        names = {names} if isinstance(names, str) else set(names)
        p = self.parent
        while p is not None:
            if p.name in names:
                return p
            p = p.parent
        return None


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Tag("[document]", {}, None)
        self.stack = [self.root]

    def handle_starttag(self, tag, attrs):
        if tag in CLOSES_P and self.stack[-1].name == "p":
            self.stack.pop()
        node = Tag(tag, {k: (v if v is not None else "") for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag, attrs):
        if tag in CLOSES_P and self.stack[-1].name == "p":
            self.stack.pop()
        node = Tag(tag, {k: (v if v is not None else "") for k, v in attrs}, self.stack[-1])
        self.stack[-1].children.append(node)

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, 0, -1):
            if self.stack[i].name == tag:
                del self.stack[i:]
                return

    def handle_data(self, data):
        kids = self.stack[-1].children
        if kids and isinstance(kids[-1], str):
            kids[-1] += data
        else:
            kids.append(data)


def parse(html_text: str) -> Tag:
    b = _Builder()
    b.feed(html_text)
    b.close()
    return b.root
