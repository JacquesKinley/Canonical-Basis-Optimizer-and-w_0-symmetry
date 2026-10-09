"""
Evacuation-adapted edges in the weak Bruhat graph on standard tableaux.

This file follows the convention of Armon--Halverson: the weak Bruhat
graph B^lambda has vertex set SYT(lambda), and T and s_i(T) are joined by
an edge labelled s_i whenever swapping i and i+1 gives another standard
tableau.

The evacuation-adapted subgraph gamma^lambda is the edge-induced subgraph
consisting of the edges

    {T, s_i(T)}  such that
    {ev(T), ev(s_i(T))} is an edge labelled s_{n-i}.

Consequently, isolated tableaux are NOT vertices of gamma^lambda.  They are
recorded separately, since in the hybrid-basis construction they contribute
the identity block.

Typical Sage use
----------------

    load("EvacuationAdaptedBruhatGraph.sage")

    D = evacuation_adapted_components((3, 2))
    D.print_adapted_subgraph()
    D.print_components()
    D.print_evacuation_action_on_components()

    # Show all components together as one simple labelled graph.
    D.show_graph()

    # Reliable fallback when a terminal has no graphics display backend.
    D.save_graph("evacuation_adapted_graph.png")

    # Component-size distribution and individual counts.
    D.size()       # for example, [2, 2, 2, 5, 5]
    D.size(2)      # number of two-vertex components
    D.size(5)      # number of five-vertex components
    D.dimensions() # distinct dimensions, for example [2, 5]

    # Draw gamma^(3,2), with vertices T1, ..., T5 and edge labels s_i.
    D.show_adapted_subgraph()

    # Draw every nontrivial connected component separately.
    D.show_components()

    # The Sage graph objects are also directly available.
    gamma = D.adapted_graph
    components = D.components

The tableaux T1, ..., Td are always indexed in last-letter (total-index)
order: compare the row containing n, then n-1, ..., then 1.
"""

from sage.all import (
    Graph,
    Partition,
    StandardTableau,
    StandardTableaux,
    graphics_array,
    show,
)


def _normalise_partition(lam):
    """Return ``lam`` as a nonempty Sage partition."""
    try:
        lam = Partition(lam)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "lambda must be a partition, for example (5, 3)"
        ) from error

    if sum(lam) == 0:
        raise ValueError("lambda must be a nonempty partition")
    return lam


def _tableau_key(T):
    """Immutable representation of a tableau, suitable as a dictionary key."""
    return tuple(tuple(int(x) for x in row) for row in T)


def _row_word(T):
    """Return the row containing each entry, using zero-based row numbers."""
    n = int(T.size())
    row_of = [None] * (n + 1)
    for row, entries in enumerate(T):
        for x in entries:
            row_of[int(x)] = row
    return row_of


def last_letter_key(T):
    r"""
    Sorting key for last-letter (total-index) order.

    The key is the tuple of row indices containing n,n-1,...,1.  Thus the
    position of the largest entry is compared first.
    """
    row_of = _row_word(T)
    n = int(T.size())
    return tuple(row_of[i] for i in range(n, 0, -1))


def tableaux_last_letter_order(lam):
    """List ``SYT(lam)`` in last-letter (total-index) order."""
    lam = _normalise_partition(lam)
    return sorted(list(StandardTableaux(lam)), key=last_letter_key)


def _swap_adjacent_entries(T, i):
    r"""
    Return s_i(T) if it is standard, and ``None`` otherwise.

    Here s_i interchanges the entries i and i+1.  This direct implementation
    avoids any ambiguity with Sage's different symmetric-group actions on
    entries and on values.
    """
    n = int(T.size())
    i = int(i)
    if i < 1 or i >= n:
        raise ValueError("i must satisfy 1 <= i < |lambda|")

    rows = [list(row) for row in T]
    position = {}
    for row, entries in enumerate(rows):
        for column, x in enumerate(entries):
            if int(x) == i or int(x) == i + 1:
                position[int(x)] = (row, column)

    r1, c1 = position[i]
    r2, c2 = position[i + 1]
    rows[r1][c1], rows[r2][c2] = rows[r2][c2], rows[r1][c1]

    candidate = StandardTableau(rows) if _is_standard_rows(rows) else None
    return candidate


def _is_standard_rows(rows):
    """Test standardness without relying on coercion exceptions."""
    # Strict increase along rows.
    for row in rows:
        if any(row[j] >= row[j + 1] for j in range(len(row) - 1)):
            return False

    # Strict increase down columns.
    for r in range(len(rows) - 1):
        common_length = min(len(rows[r]), len(rows[r + 1]))
        for c in range(common_length):
            if rows[r][c] >= rows[r + 1][c]:
                return False
    return True


def _edge_label_text(i):
    """Readable form of the simple-transposition label."""
    return "s_%s" % int(i)


def _display_graph(G, name=None):
    """Copy an integer-labelled graph for plotting as T1 --s_i-- T2."""
    H = Graph(multiedges=False, loops=False)
    H.add_vertices(["T%s" % int(v) for v in G.vertices(sort=True)])
    for u, v, i in G.edges(sort=True, labels=True):
        H.add_edge("T%s" % int(u), "T%s" % int(v), _edge_label_text(i))
    if name is not None:
        H.name(name)
    return H


class EvacuationAdaptedBruhatData(object):
    """Container returned by :func:`evacuation_adapted_components`."""

    def __init__(
        self,
        lam,
        tableaux,
        bruhat_graph,
        adapted_spanning_graph,
        adapted_graph,
        components,
        isolated_vertices,
        evacuation,
        component_partner,
    ):
        self.partition = lam
        self.n = int(sum(lam))
        self.tableaux = tableaux
        self.bruhat_graph = bruhat_graph
        self.adapted_spanning_graph = adapted_spanning_graph
        self.adapted_graph = adapted_graph
        self.components = components
        self.isolated_vertices = isolated_vertices
        self.evacuation = evacuation
        self.component_partner = component_partner

    @property
    def gamma(self):
        """Alias for the evacuation-adapted graph gamma^lambda."""
        return self.adapted_graph

    def tableau(self, k):
        """Return T_k, where indexing starts at 1."""
        k = int(k)
        if k < 1 or k > len(self.tableaux):
            raise IndexError("tableau index out of range")
        return self.tableaux[k - 1]

    def print_tableaux(self):
        """Print T1,...,Td in last-letter order."""
        print("Tableaux in last-letter order:")
        for k, T in enumerate(self.tableaux, start=1):
            print("T%s = %s" % (k, T))

    def print_summary(self):
        """Print the basic graph and component statistics."""
        print("lambda =", tuple(self.partition))
        print("number of tableaux:", len(self.tableaux))
        print("number of weak Bruhat edges:", self.bruhat_graph.num_edges())
        print("number of evacuation-adapted edges:", self.adapted_graph.num_edges())
        print("number of evacuation-adapted components:", len(self.components))
        print("component sizes:", self.component_sizes)
        print("component-size distribution:", self.size_distribution())
        print("number of tableaux outside gamma:", len(self.isolated_vertices))

    @property
    def component_sizes(self):
        """Return ``[|X_1|,...,|X_s|]`` in component order."""
        return [int(C.num_verts()) for C in self.components]

    def size(self, component_size=None):
        r"""
        Return component dimensions, or count one specified dimension.

        - ``D.size()`` returns ``[|X_1|,...,|X_s|]`` in component order;
        - ``D.size(r)`` returns the number of components having exactly ``r``
          vertices.

        Thus ``D.size(2)`` and ``D.size(5)`` count the two-dimensional and
        five-dimensional connected components used in the hybrid construction.
        """
        if component_size is None:
            return list(self.component_sizes)

        component_size = int(component_size)
        if component_size < 1:
            raise ValueError("component_size must be a positive integer")
        return self.component_sizes.count(component_size)

    def dimensions(self):
        """Return the sorted list of distinct component dimensions."""
        return sorted(set(self.component_sizes))

    def size_distribution(self):
        """Return ``{dimension: multiplicity}`` for all component dimensions."""
        distribution = {}
        for r in self.component_sizes:
            distribution[r] = distribution.get(r, 0) + 1
        return dict(sorted(distribution.items()))

    def print_component_sizes(self):
        """Print how many components occur with each number of vertices."""
        distribution = self.size_distribution()
        if not distribution:
            print("There are no evacuation-adapted connected components.")
            return

        print("Evacuation-adapted component-size distribution:")
        for r, multiplicity in distribution.items():
            noun = "component" if multiplicity == 1 else "components"
            print("  size %s: %s %s" % (r, multiplicity, noun))

    def print_adapted_subgraph(self, print_tableaux=True):
        """Print the vertices and labelled edges of gamma^lambda."""
        print("Evacuation-adapted subgraph gamma^%s" % (tuple(self.partition),))
        print("  vertices:", ["T%s" % v for v in self.adapted_graph.vertices(sort=True)])
        print("  edges:")
        edges = self.adapted_graph.edges(sort=True, labels=True)
        if not edges:
            print("    (none)")
        else:
            for u, v, i in edges:
                print("    T%s --%s-- T%s" % (u, _edge_label_text(i), v))

        if self.isolated_vertices:
            print(
                "  tableaux outside gamma (isolated after the edge filter):",
                ["T%s" % v for v in self.isolated_vertices],
            )
        else:
            print("  tableaux outside gamma: none")

        if print_tableaux:
            print()
            self.print_tableaux()

    def print_components(self, print_tableaux=True, print_edges=True):
        """Print the disjoint connected components X_1,...,X_s."""
        if not self.components:
            print("Connected-component decomposition: gamma has no vertices.")
            print("  There are no evacuation-adapted edges.")
            return

        print(
            "Connected-component decomposition: gamma^%s = X_1 sqcup ... sqcup X_%s"
            % (tuple(self.partition), len(self.components))
        )
        for a, component in enumerate(self.components, start=1):
            vertices = component.vertices(sort=True)
            print("X_%s:" % a)
            print("  vertices:", ["T%s" % v for v in vertices])

            if print_tableaux:
                for v in vertices:
                    print("    T%s = %s" % (v, self.tableau(v)))

            if print_edges:
                print("  edges:")
                for u, v, i in component.edges(sort=True, labels=True):
                    print("    T%s --%s-- T%s" % (u, _edge_label_text(i), v))

    def print_evacuation_action_on_components(self):
        """Print the involution induced by evacuation on the components."""
        if not self.components:
            print("There are no nontrivial evacuation-adapted components.")
            return

        print("Evacuation action on connected components:")
        for a, b in enumerate(self.component_partner, start=1):
            print("  ev(X_%s) = X_%s" % (a, b))

    def plot_adapted_subgraph(self, layout="spring", **plot_options):
        """Return a Sage graphics object for gamma^lambda."""
        H = _display_graph(
            self.adapted_graph,
            name="Evacuation-adapted subgraph for lambda=%s" % (tuple(self.partition),),
        )
        options = {
            "layout": layout,
            "edge_labels": True,
            "vertex_labels": True,
            "vertex_size": 650,
        }
        options.update(plot_options)
        return H.plot(**options)

    def show_adapted_subgraph(self, layout="spring", **plot_options):
        """Display gamma^lambda."""
        show(self.plot_adapted_subgraph(layout=layout, **plot_options))

    def plot_graph(self, layout="spring", ncols=3, **plot_options):
        r"""
        Return all components of gamma^lambda in one uncrowded graphics array.

        Each connected component is laid out independently in its own panel.
        This prevents the large components from being compressed when Sage
        packs a disconnected graph into one long horizontal strip.
        """
        if not self.components:
            return self.plot_adapted_subgraph(layout=layout, **plot_options)

        ncols = max(1, min(int(ncols), len(self.components)))
        pictures = [
            self.plot_component(a, layout=layout, **plot_options)
            for a in range(1, len(self.components) + 1)
        ]
        return graphics_array(pictures, ncols=ncols)

    def show_graph(
        self,
        layout="spring",
        ncols=3,
        figsize=None,
        **plot_options
    ):
        r"""Display all components in a spacious multi-row grid."""
        picture = self.plot_graph(
            layout=layout, ncols=ncols, **plot_options
        )
        if figsize is None:
            columns = min(max(1, int(ncols)), max(1, len(self.components)))
            rows = max(1, (len(self.components) + columns - 1) // columns)
            figsize = (5 * columns, 4.5 * rows)
        show(picture, figsize=figsize)

    def save_graph(
        self,
        filename="evacuation_adapted_graph.png",
        layout="spring",
        ncols=3,
        figsize=None,
        dpi=200,
        **plot_options
    ):
        r"""
        Save the complete evacuation-adapted graph to an image file.

        This is useful in a terminal or WSL session without an inline graphics
        backend.  The filename extension selects the format; PNG, PDF and SVG
        are convenient choices.
        """
        picture = self.plot_graph(
            layout=layout, ncols=ncols, **plot_options
        )
        if figsize is None:
            columns = min(max(1, int(ncols)), max(1, len(self.components)))
            rows = max(1, (len(self.components) + columns - 1) // columns)
            figsize = (5 * columns, 4.5 * rows)
        picture.save(filename, dpi=int(dpi), figsize=figsize)
        print("Saved evacuation-adapted graph to %s" % filename)
        return filename

    def plot_component(self, component_number, layout="spring", **plot_options):
        """Return a Sage graphics object for one component X_a."""
        a = int(component_number)
        if a < 1 or a > len(self.components):
            raise IndexError("component_number must be between 1 and %s" % len(self.components))

        H = _display_graph(self.components[a - 1], name="X_%s" % a)
        options = {
            "layout": layout,
            "iterations": 500,
            "edge_labels": True,
            "vertex_labels": True,
            "vertex_size": 400,
            "title": "X_%s (size %s)" % (a, self.components[a - 1].num_verts()),
        }
        options.update(plot_options)
        return H.plot(**options)

    def show_components(self, layout="spring", **plot_options):
        """Display every component X_a separately."""
        for a in range(1, len(self.components) + 1):
            print("X_%s" % a)
            show(self.plot_component(a, layout=layout, **plot_options))


def evacuation_adapted_components(
    lam,
    print_result=True,
    show_graph=False,
    verify=True,
):
    r"""
    Construct the evacuation-adapted subgraph of the weak Bruhat graph.

    INPUT:

    - ``lam`` -- a partition, for example ``(5,3)`` or ``[5,3]``;
    - ``print_result`` -- if ``True`` (default), print the adapted subgraph
      and its connected components;
    - ``show_graph`` -- if ``True``, display the adapted subgraph;
    - ``verify`` -- if ``True`` (default), check the evacuation symmetry and
      the component decomposition.

    OUTPUT:

    An :class:`EvacuationAdaptedBruhatData` object.  Its most useful fields are

    - ``.bruhat_graph``: the full weak Bruhat graph;
    - ``.adapted_graph`` or ``.gamma``: gamma^lambda, with isolated tableaux
      excluded;
    - ``.components``: the ordered list [X_1,...,X_s] of component graphs;
    - ``.isolated_vertices``: tableaux outside gamma^lambda;
    - ``.evacuation``: dictionary k |--> ell when ev(T_k)=T_ell.

    Components are ordered by their least tableau in last-letter order.
    """
    lam = _normalise_partition(lam)
    n = int(sum(lam))

    # Fix the project-wide tableau order before constructing any graph data.
    tableaux = tableaux_last_letter_order(lam)
    d = len(tableaux)
    index_of = {_tableau_key(T): k for k, T in enumerate(tableaux, start=1)}

    # Full weak Bruhat graph.  Vertices are the integer indices T_1,...,T_d;
    # edge labels are the integers i representing s_i.
    bruhat = Graph(multiedges=False, loops=False)
    bruhat.add_vertices(range(1, d + 1))

    for u, T in enumerate(tableaux, start=1):
        for i in range(1, n):
            U = _swap_adjacent_entries(T, i)
            if U is None:
                continue
            v = index_of[_tableau_key(U)]
            if u < v:
                bruhat.add_edge(u, v, i)

    # Evacuation permutation on tableau indices.
    evacuation = {}
    for u, T in enumerate(tableaux, start=1):
        evT = StandardTableau(T.evacuation())
        evacuation[u] = index_of[_tableau_key(evT)]

    # Retain exactly the edges satisfying the user's pointwise condition.
    adapted_spanning = Graph(multiedges=False, loops=False)
    adapted_spanning.add_vertices(range(1, d + 1))

    for u, v, i in bruhat.edges(sort=True, labels=True):
        ev_u = evacuation[u]
        ev_v = evacuation[v]
        reflected_label = n - int(i)
        if (
            bruhat.has_edge(ev_u, ev_v)
            and bruhat.edge_label(ev_u, ev_v) == reflected_label
        ):
            adapted_spanning.add_edge(u, v, i)

    # gamma^lambda is edge-induced: remove all isolated vertices.
    incident_vertices = sorted(
        v for v in adapted_spanning.vertices(sort=False)
        if adapted_spanning.degree(v) > 0
    )
    adapted = adapted_spanning.subgraph(incident_vertices)
    isolated_vertices = sorted(set(range(1, d + 1)) - set(incident_vertices))

    # Sage does not promise the component order needed by the construction,
    # so impose a deterministic last-letter order explicitly.
    component_vertex_sets = [
        sorted(C) for C in adapted.connected_components(sort=False)
    ]
    component_vertex_sets.sort(key=lambda C: C[0])
    components = [adapted.subgraph(C) for C in component_vertex_sets]

    component_of = {}
    for a, C in enumerate(component_vertex_sets, start=1):
        for v in C:
            component_of[v] = a

    component_partner = []
    for C in component_vertex_sets:
        image_components = {component_of[evacuation[v]] for v in C}
        if len(image_components) != 1:
            raise RuntimeError(
                "internal error: evacuation does not send a component to one component"
            )
        component_partner.append(next(iter(image_components)))

    data = EvacuationAdaptedBruhatData(
        lam=lam,
        tableaux=tableaux,
        bruhat_graph=bruhat,
        adapted_spanning_graph=adapted_spanning,
        adapted_graph=adapted,
        components=components,
        isolated_vertices=isolated_vertices,
        evacuation=evacuation,
        component_partner=component_partner,
    )

    if verify:
        _verify_evacuation_adapted_data(data)

    if print_result:
        data.print_summary()
        print()
        data.print_adapted_subgraph(print_tableaux=False)
        print()
        data.print_components(print_tableaux=True, print_edges=True)
        print()
        data.print_evacuation_action_on_components()

    if show_graph:
        data.show_adapted_subgraph()

    return data


def _verify_evacuation_adapted_data(data):
    """Internal consistency checks for the constructed graphs."""
    n = data.n
    d = len(data.tableaux)

    # Evacuation must be an involution on all tableaux.
    for u in range(1, d + 1):
        if data.evacuation[data.evacuation[u]] != u:
            raise RuntimeError("evacuation failed the involution check at T%s" % u)

    # Every adapted colour-i edge must map to a colour-(n-i) adapted edge.
    for u, v, i in data.adapted_graph.edges(sort=True, labels=True):
        ev_u = data.evacuation[u]
        ev_v = data.evacuation[v]
        expected_label = n - int(i)
        if not data.adapted_spanning_graph.has_edge(ev_u, ev_v):
            raise RuntimeError("adapted edge failed evacuation closure")
        if data.adapted_spanning_graph.edge_label(ev_u, ev_v) != expected_label:
            raise RuntimeError("adapted edge failed the reflected-label check")

    # Components must be pairwise disjoint and exhaust precisely V(gamma).
    component_vertices = []
    for C in data.components:
        component_vertices.extend(C.vertices(sort=False))
    if len(component_vertices) != len(set(component_vertices)):
        raise RuntimeError("connected components are not pairwise disjoint")
    if set(component_vertices) != set(data.adapted_graph.vertices(sort=False)):
        raise RuntimeError("connected components do not exhaust gamma")

    # The induced action on components must itself be an involution.
    for a, b in enumerate(data.component_partner, start=1):
        if data.component_partner[b - 1] != a:
            raise RuntimeError("evacuation action on components is not an involution")

    return True


def evacuation_adapted_self_test(verbose=True):
    r"""
    Run regression tests on lambda=(3,2) and lambda=(4,3).

    The expected output is

        X_1 = {T_1,T_2},     with edge label s_2,
        X_2 = {T_4,T_5},     with edge label s_3,

    while T_3 is isolated after the edge filter and evacuation exchanges
    X_1 and X_2.
    """
    data = evacuation_adapted_components(
        (3, 2), print_result=False, show_graph=False, verify=True
    )

    component_vertices = [C.vertices(sort=True) for C in data.components]
    adapted_edges = data.adapted_graph.edges(sort=True, labels=True)

    assert component_vertices == [[1, 2], [4, 5]]
    assert adapted_edges == [(1, 2, 2), (4, 5, 3)]
    assert data.isolated_vertices == [3]
    assert data.evacuation == {1: 4, 2: 5, 3: 3, 4: 1, 5: 2}
    assert data.component_partner == [2, 1]
    assert data.size() == [2, 2]
    assert data.size(2) == 2
    assert data.dimensions() == [2]
    assert data.size_distribution() == {2: 2}

    # This example simultaneously tests two- and five-dimensional components.
    data_43 = evacuation_adapted_components(
        (4, 3), print_result=False, show_graph=False, verify=True
    )
    assert data_43.size() == [5, 2, 2, 5]
    assert data_43.size(2) == 2
    assert data_43.size(5) == 2
    assert data_43.dimensions() == [2, 5]
    assert data_43.size_distribution() == {2: 2, 5: 2}

    if verbose:
        print("Evacuation-adapted Bruhat graph self-tests passed for lambda=(3,2) and (4,3).")
    return True


# ---------------------------------------------------------------------------
# Quick examples (uncomment after loading the file in Sage)
# ---------------------------------------------------------------------------
# D32 = evacuation_adapted_components((3, 2))
# D32.show_adapted_subgraph()
# evacuation_adapted_self_test()
#
# D54 = evacuation_adapted_components((5, 4), print_result=False)
# D54.print_components()
# D54.print_evacuation_action_on_components()
# D54.show_components()
