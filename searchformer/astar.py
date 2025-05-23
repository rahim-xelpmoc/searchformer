# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import pickle
import random
from abc import abstractmethod
from dataclasses import dataclass
from heapq import heapify, heappop, heappush
from typing import Any, Dict, List, Optional, Sequence


def sokoban_state_to_pretty_string(state: List[List[str]]) -> str:
    return "\n".join(["".join(row) for row in state])


class AStarState:
    """This class implements a state or node in A* search. A specific
    implementation of A* search for a task should implement this class as a
    sub-class.
    """

    def __init__(
        self,
        parent: Optional["AStarState"] = None,
        deterministic: bool = True,
    ):
        """Instatiates A* state.

        Args:
            parent (Optional[&quot;AStarState&quot;], optional): Parent node.
                At start this is set to None. Defaults to None.
            deterministic (bool, optional): If false, the returned child
                nodes are shuffled. This flag is often also used in the
                __le__ and __lt__ methods to randomize A*'s search
                dynamics. Defaults to True.
        """
        self.parent = parent
        self.deterministic = deterministic
        self.path = ""

    @property
    @abstractmethod
    def state(self) -> Dict[str, Any]:
        raise NotImplementedError()

    @abstractmethod
    def _get_children(self) -> List["AStarState"]:
        raise NotImplementedError()

    @property
    def children(self) -> List["AStarState"]:
        child_node_list = self._get_children()
        if not self.deterministic:
            random.shuffle(child_node_list)
        return child_node_list

    @property
    @abstractmethod
    def heuristic(self) -> float:
        raise NotImplementedError()

    @property
    @abstractmethod
    def cost_from_start(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def __hash__(self) -> int:
        """The hash is used to establish identity between different states. This
        function is implemented specifically to control the search behaviour of A* and
        integrate an implicit ordering of child nodes.

        Raises:
            NotImplementedError: If not implemented.

        Returns:
            int: Hash value of state.
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def is_goal(self) -> bool:
        raise NotImplementedError()

    @property
    def cost(self) -> float:
        return self.heuristic + self.cost_from_start

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AStarState):
            return False
        return hash(self) == hash(other)

    def __lt__(self, other: "AStarState") -> bool:
        if self.cost == other.cost and self.deterministic:
            return hash(self) < hash(other)
        elif self.cost == other.cost and not self.deterministic:
            return random.choice([False, True])
        else:
            return self.cost < other.cost

    def __le__(self, other: "AStarState") -> bool:
        if self.cost == other.cost and self.deterministic:
            return hash(self) < hash(other)
        elif self.cost == other.cost and not self.deterministic:
            return random.choice([False, True])
        else:
            return self.cost < other.cost


@dataclass
class TraceStep:
    """Data class to store an A* execution trace step."""

    action: str
    state: Dict[str, Any]
    cost_from_start: int
    heuristic: int
    path: str

    def to_dict(self) -> Dict[str, Any]:
        return dict(
            action=self.action,
            state=self.state,
            cost_from_start=self.cost_from_start,
            heuristic=self.heuristic,
            path=self.path,
        )

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "TraceStep":
        return TraceStep(**d)


class CreateStep(TraceStep):
    def __init__(self, **kvargs):
        super().__init__(action="create", **kvargs)


class CloseStep(TraceStep):
    def __init__(self, **kvargs):
        super().__init__(action="close", **kvargs)


class PlanStep(TraceStep):
    def __init__(self, **kvargs):
        super().__init__(action="plan", **kvargs)


class AStarCannotSolveTaskException(Exception):
    def __init__(self, trace: Optional[List[TraceStep]] = None):
        super().__init__()
        self.trace = trace


def astar(start_state: AStarState) -> Sequence[TraceStep]:
    """A* implementation used for generating execution trace datasets.

    The start state (or node) is provided as input and expanded until an
    optimal plan is found.

    Args:
        start_state (AStarState): Start state.

    Raises:
        AStarCannotSolveTaskException: If no feasible plan is found
            and the goal is not reached.

    Returns:
        Sequence[TraceStep]: Sequence of execution trace steps.
    """
    open_heap: List[AStarState] = []
    open_dict: Dict[AStarState, AStarState] = {}
    closed_dict: Dict[AStarState, AStarState] = {}
    log: List[TraceStep] = []

    curr_node = start_state
    heappush(open_heap, curr_node)
    open_dict[curr_node] = curr_node
    log.append(
        CreateStep(
            state=curr_node.state,
            cost_from_start=curr_node.cost_from_start,
            heuristic=curr_node.heuristic,
            path=curr_node.path,
        )
    )

    while len(open_heap) > 0:
        curr_node = heappop(open_heap)
        del open_dict[curr_node]
        closed_dict[curr_node] = curr_node
        log.append(
            CloseStep(
                state=curr_node.state,
                cost_from_start=curr_node.cost_from_start,
                heuristic=curr_node.heuristic,
                path=curr_node.path,
            )
        )
        if curr_node.cost == float("inf"):
            raise AStarCannotSolveTaskException(log)
        if curr_node.is_goal:
            break

        for child_node in curr_node.children:
            if child_node in open_dict.keys():
                if open_dict[child_node].cost <= child_node.cost:
                    continue
                else:
                    # This deletion is necessary because the hash is a function
                    # of the state, not the cost of the node. If there a lower
                    # cost value is computed for the same state, the following
                    # will prevent adding multiple nodes with the same state
                    # but different costs to the heap.
                    open_heap.remove(child_node)
                    heapify(open_heap)
                    del open_dict[child_node]
            if child_node in closed_dict.keys():
                if closed_dict[child_node].cost <= child_node.cost:
                    continue

            heappush(open_heap, child_node)
            open_dict[child_node] = child_node
            log.append(
                CreateStep(
                    state=child_node.state,
                    cost_from_start=child_node.cost_from_start,
                    heuristic=child_node.heuristic,
                    path=child_node.path,
                )
            )
    if not curr_node.is_goal:
        raise AStarCannotSolveTaskException(log)

    path: List[AStarState] = [curr_node]
    node = curr_node.parent
    while node is not None:
        path.insert(0, node)
        node = node.parent
    for node in path:
        log.append(
            PlanStep(
                state=node.state,
                cost_from_start=node.cost_from_start,
                heuristic=node.heuristic,
                path=node.path,
            )
        )

    return log


def astar_verbose(start_state: AStarState) -> Sequence[TraceStep]:
    """A* implementation used for generating execution trace datasets.

    The start state (or node) is provided as input and expanded until an
    optimal plan is found.

    Args:
        start_state (AStarState): Start state.

    Raises:
        AStarCannotSolveTaskException: If no feasible plan is found
            and the goal is not reached.

    Returns:
        Sequence[TraceStep]: Sequence of execution trace steps.
    """
    open_heap: List[AStarState] = []
    open_heap2 = []
    open_dict: Dict[AStarState, AStarState] = {}
    open_dict2 = {}
    closed_dict: Dict[AStarState, AStarState] = {}
    closed_dict2 = {}
    log: List[TraceStep] = []
    print("root state:")
    print(sokoban_state_to_pretty_string(state=start_state.state["state"]))
    print("node_id:", start_state.path)
    print("cost:", start_state.cost)
    print("heap:", open_heap2)
    # print("open_dict:", open_dict2)
    # print("closed_dict:", closed_dict2)
    print("currently min heap is empty. Pushing root state to the heap")
    print("Action: push_heap(start_state)")
    print("observation: done")
    print(f"Action: push_open(node_id='{start_state.path}')")
    print("observation: done")
    curr_node = start_state
    heappush(open_heap, curr_node)
    heappush(open_heap2, (curr_node.cost, curr_node.path))
    open_dict[curr_node] = curr_node
    open_dict2[curr_node.path] = curr_node.cost
    print("heap:", open_heap2)
    # print("open_dict:", open_dict2)
    log.append(
        CreateStep(
            state=curr_node.state,
            cost_from_start=curr_node.cost_from_start,
            heuristic=curr_node.heuristic,
            path=curr_node.path,
        )
    )

    print("initializing search")
    while len(open_heap) > 0:
        curr_node = heappop(open_heap)
        heappop(open_heap2)
        print("selecting the node from the heap with lowest cost")
        print(f"Action: fetch_node(node_id={curr_node.path})")
        print("observation:")
        print("current state:")
        print(sokoban_state_to_pretty_string(state=curr_node.state["state"]))
        print("node_id:", curr_node.path)
        print("cost:", curr_node.cost)
        del open_dict[curr_node]
        del open_dict2[curr_node.path]
        closed_dict[curr_node] = curr_node
        print("pushing node to the closed list")
        print(f"Action: push_closed(node_id={curr_node.path})")
        print("observation: done")
        closed_dict2[curr_node.path] = curr_node.cost
        log.append(
            CloseStep(
                state=curr_node.state,
                cost_from_start=curr_node.cost_from_start,
                heuristic=curr_node.heuristic,
                path=curr_node.path,
            )
        )
        if curr_node.cost == float("inf"):
            print("current node cost is infinity")
            print("no feasible plan found")
            # pickle.dump(log, open("./log.pkl", "wb"))
            raise AStarCannotSolveTaskException(log)
        if curr_node.is_goal:
            print("current node is the goal")
            print("goal reached")
            break
        print("exploring child nodes")
        for child_node in curr_node.children:
            print("current child:")
            # print(
            #     sokoban_state_to_pretty_string(state=child_node.state["state"])
            # )
            print("node_id:", child_node.path)
            print("cost:", child_node.cost)
            print("checks:")
            print(f"Action: check_open(node_id={child_node.path})")
            if child_node in open_dict.keys():
                print("observation: child node is already in heap")
                if open_dict[child_node].cost <= child_node.cost:
                    print(
                        "the cost of the child in open_dict is less than the child itself so continuing."
                    )
                    continue
                else:
                    # This deletion is necessary because the hash is a function
                    # of the state, not the cost of the node. If there a lower
                    # cost value is computed for the same state, the following
                    # will prevent adding multiple nodes with the same state
                    # but different costs to the heap.
                    # print(
                    #     "child node is already in heap with more cost value. Removing it from the heap."
                    # )
                    print(
                        "the cost of the child in open_dict is more than the child itself so removing it from the heap."
                    )
                    print(f"Action: remove_heap(node_id={child_node.path})")
                    print("observation: done")
                    open_heap.remove(child_node)
                    open_heap2.remove((child_node.cost, child_node.path))
                    heapify(open_heap)
                    heapify(open_heap2)
                    del open_dict[child_node]
                    del open_dict2[child_node.path]
            print("child is not in the open_dict")
            print(f"Action: check_closed(node_id={child_node.path})")
            if child_node in closed_dict.keys():
                print("observation: child is in the closed_dict")
                if closed_dict[child_node].cost <= child_node.cost:
                    print(
                        "the cost of the child in closed_dict is less than the child itself so continuing"
                    )
                    continue

            print("observation: child is not in the open_dict and not in the closed dict.")
            heappush(open_heap, child_node)
            heappush(open_heap2, (child_node.cost, child_node.path))
            print("adding child node to the heap")
            print(f"Action: push_heap(node_id={child_node.path})")
            open_dict[child_node] = child_node
            open_dict2[child_node.path] = child_node.cost
            log.append(
                CreateStep(
                    state=child_node.state,
                    cost_from_start=child_node.cost_from_start,
                    heuristic=child_node.heuristic,
                    path=child_node.path,
                )
            )
            print("observation: done")
        print("heap:", open_heap2)

        # print("open_dict:", open_dict2)
        # print("closed_dict:", closed_dict2)
    if not curr_node.is_goal:
        print("no feasible plan found")
        # pickle.dump(log, open("./log.pkl", "wb"))
        raise AStarCannotSolveTaskException(log)

    path: List[AStarState] = [curr_node]
    node = curr_node.parent
    while node is not None:
        path.insert(0, node)
        node = node.parent
    for node in path:
        log.append(
            PlanStep(
                state=node.state,
                cost_from_start=node.cost_from_start,
                heuristic=node.heuristic,
                path=node.path,
            )
        )

    return log
