"""Focused contract tests for the repository Markdown quality gate."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts import check_markdown_docs


class MarkdownDocsCheckerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self._temporary_directory.name).resolve()

    def tearDown(self) -> None:
        self._temporary_directory.cleanup()

    def _write(self, relative_path: str, content: str) -> Path:
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path

    def _errors(self, relative_path: str) -> list[str]:
        return check_markdown_docs._validate_file(
            self.root,
            self.root / relative_path,
            {},
        )

    def test_local_inline_image_reference_and_encoded_links_resolve(self) -> None:
        self._write("docs/guide (v2).md", "# Install & Run\n")
        self._write("assets/graph.svg", "<svg />")
        self._write(
            "README.md",
            """# Home

[Guide](<docs/guide%20%28v2%29.md#install--run>)

![Graph](assets/graph.svg)

[Reference guide][guide]

[guide]: <docs/guide%20%28v2%29.md#install--run>
""",
        )

        self.assertEqual(self._errors("README.md"), [])

    def test_missing_and_repository_escaping_targets_are_rejected(self) -> None:
        self._write(
            "README.md",
            """# Home

[Missing](missing.md)

[Outside](../outside.md)
""",
        )

        errors = self._errors("README.md")

        self.assertEqual(len(errors), 2)
        self.assertIn("missing local link target: missing.md", errors[0])
        self.assertIn("link escapes repository: ../outside.md", errors[1])

    def test_balanced_parentheses_are_supported_in_bare_link_destinations(
        self,
    ) -> None:
        self._write("guide_(v2).md", "# Guide\n")
        self._write(
            "README.md",
            """# Home

[Balanced](guide_(v2).md)

[Escaped](guide_\\(v2\\).md)
""",
        )

        self.assertEqual(self._errors("README.md"), [])

    def test_github_anchors_cover_punctuation_duplicates_and_slug_collisions(
        self,
    ) -> None:
        guide = self._write(
            "guide.md",
            """# API & CLI
## API & CLI
### Ontology alignment and the SHACL/provenance promotion gate
### Release (`v1.0`) ###
### snake_case
### R&D &amp; QA
### Foo
### Foo-1
### Foo
```markdown
## Hidden heading
~~~
## Still hidden
```
""",
        )

        self.assertEqual(
            check_markdown_docs._anchors(guide),
            {
                "api--cli",
                "api--cli-1",
                "ontology-alignment-and-the-shaclprovenance-promotion-gate",
                "release-v10",
                "snake_case",
                "rd--qa",
                "foo",
                "foo-1",
                "foo-2",
            },
        )

    def test_valid_duplicate_and_punctuation_anchors_resolve(self) -> None:
        self._write(
            "guide.md",
            """# API & CLI
## API & CLI
## Ontology alignment and the SHACL/provenance promotion gate
""",
        )
        self._write(
            "README.md",
            """# Home

[First](guide.md#api--cli)

[Second](guide.md#api--cli-1)

[Promotion](guide.md#ontology-alignment-and-the-shaclprovenance-promotion-gate)
""",
        )

        self.assertEqual(self._errors("README.md"), [])

    def test_missing_heading_anchor_is_rejected(self) -> None:
        self._write("guide.md", "# Existing heading\n")
        self._write("README.md", "# Home\n\n[Missing](guide.md#not-present)\n")

        self.assertEqual(
            self._errors("README.md"),
            [
                "README.md:3: missing heading anchor #not-present in guide.md",
            ],
        )

    def test_heading_fragments_are_case_sensitive(self) -> None:
        self._write("guide.md", "# Existing heading\n")
        self._write(
            "README.md",
            "# Home\n\n[Wrong case](guide.md#Existing-heading)\n",
        )

        self.assertEqual(
            self._errors("README.md"),
            [
                "README.md:3: missing heading anchor #Existing-heading "
                "in guide.md",
            ],
        )

    def test_fenced_links_and_headings_are_ignored_until_matching_close(
        self,
    ) -> None:
        document = self._write(
            "README.md",
            """# Visible

````markdown
[Example only](missing.md)
## Hidden heading
~~~
[Still example only](also-missing.md)
````
""",
        )

        self.assertEqual(self._errors("README.md"), [])
        self.assertEqual(check_markdown_docs._anchors(document), {"visible"})

    def test_shorter_or_different_fence_does_not_close_a_code_block(self) -> None:
        self._write(
            "README.md",
            """# Home

````markdown
~~~
```
""",
        )

        self.assertEqual(
            self._errors("README.md"),
            ["README.md: unclosed fenced code block"],
        )

    def test_longer_matching_fence_closes_a_code_block(self) -> None:
        self._write(
            "README.md",
            """# Home

```markdown
[Example only](missing.md)
````
""",
        )

        self.assertEqual(self._errors("README.md"), [])

    def test_consecutive_prose_lines_are_reported_as_a_hard_wrap(self) -> None:
        self._write(
            "README.md",
            """# Home

One logical paragraph starts here
and continues on another source line.
""",
        )

        self.assertEqual(
            self._errors("README.md"),
            [
                "README.md:4: unintended hard wrap after line 3; "
                "keep one logical paragraph on one source line",
            ],
        )

    def test_separate_paragraphs_and_structural_blocks_are_not_hard_wraps(
        self,
    ) -> None:
        self._write(
            "README.md",
            """# Home

First paragraph.

Second paragraph.

- First item
- Second item

| Column |
| --- |
| Value |

> One-line callout.

```text
code may use
multiple lines
```
""",
        )

        self.assertEqual(self._errors("README.md"), [])

    def test_navigation_reachability_follows_files_directories_and_references(
        self,
    ) -> None:
        self._write(
            "README.md",
            """# Home

[Guide](docs/)
""",
        )
        self._write(
            "docs/README.md",
            """# Guide

[Details][details]

[details]: details.md
""",
        )
        self._write("docs/details.md", "# Details\n")

        files = check_markdown_docs._documentation_files(self.root)

        self.assertEqual(
            check_markdown_docs._unreachable_documentation_files(self.root, files),
            [],
        )

    def test_navigation_reachability_reports_orphan_pages(self) -> None:
        self._write("README.md", "# Home\n")
        orphan = self._write("docs/orphan.md", "# Orphan\n")

        files = check_markdown_docs._documentation_files(self.root)

        self.assertEqual(
            check_markdown_docs._unreachable_documentation_files(self.root, files),
            [orphan],
        )


if __name__ == "__main__":
    unittest.main()
