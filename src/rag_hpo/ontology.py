from functools import lru_cache

import pronto
from pronto.ontology import _OntologyTerms

from rag_hpo.constants import ROOT_ID


class Ontology:
    def __init__(self, ontology: pronto.Ontology, root_id=ROOT_ID):
        self._inner = ontology
        self._root_id = root_id

    def root_id(self):
        return self._root_id

    def terms(self) -> _OntologyTerms:
        return self._inner.terms()

    def root_to_node_paths(self, term_id: str) -> list[str]:
        """
        Return all paths from ROOT_ID → ... → hp_id.
        Each path is a list of HP_ID strings.
        Iterative DFS; avoids cycles by tracking nodes on the current DFS path.
        """
        paths = list()
        lineage_memo = dict()

        on_path = set()  # nodes currently being expanded (cycle guard)
        stack = [(term_id, False)]  # (node, parents_already_pushed)

        while stack:
            node, expanded = stack.pop()

            if node in lineage_memo:
                continue

            # base case: we hit the root
            if node == self._root_id:
                lineage_memo[node] = [[self._root_id]]
                continue

            parents = self.get_direct_parents(node)
            if not parents:
                # orphan: just attach root + self
                lineage_memo[node] = [[self._root_id, node]]
                continue

            if not expanded:
                if node in on_path:  # cycle
                    continue
                on_path.add(node)
                # Revisit this node after all its parents are resolved
                stack.append((node, True))
                for p in parents:
                    if p not in lineage_memo and p not in on_path:
                        stack.append((p, False))
            else:
                # All parents resolved; a parent missing from the memo was
                # cut off by the cycle guard, so it contributes no paths.
                paths = []
                for p in parents:
                    for ppath in lineage_memo.get(p, []):
                        paths.append(ppath + [node])
                lineage_memo[node] = paths
                on_path.discard(node)

        return paths

    @lru_cache(maxsize=500)
    def get_direct_parents(self, term_id: str) -> list[str]:
        """Returns a dictionary mapping ontology terms to their direct parents."""
        term = self._inner.get_term(term_id)
        return [p.id for p in term.superclasses(distance=1)]

    @lru_cache(maxsize=500)
    def get_label(self, term_id: str) -> str | None:
        """Returns the label of the term if it exists, otherwise None."""
        term = self._inner.get_term(term_id)
        return term.name
