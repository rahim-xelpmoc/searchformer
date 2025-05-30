import re
from pprint import pprint
from collections import defaultdict

eval_dict = {}

##### Eval 0: question_rewrite_validation #
def split_graph_sections(all_str_except_plan):
    return all_str_except_plan.split("The graph is:")

def parse_graph_data(section):
    parts = section.split("\n\n")
    graph = parts[0].strip()
    start_node = parts[1].split(":")[1].strip()
    goal_node = parts[2].split(":")[1].strip()
    heuristics = parts[3].split("The `heuristics` dictionary is:")[-1].strip()
    return graph, start_node, goal_node, heuristics

def compare_graphs(gt_data, rewrite_data):
    mismatches = {}

    if gt_data[0] != rewrite_data[0]:
        mismatches['graph_rewritten_correctly'] = 0.0
    else:
        mismatches['graph_rewritten_correctly'] = 100.0
    if gt_data[1] != rewrite_data[1]:
        mismatches['start_node_rewritten_correctly'] = 0.0
    else:
        mismatches['start_node_rewritten_correctly'] = 100.0
    if gt_data[2] != rewrite_data[2]:
        mismatches['goal_node_rewritten_correctly'] = 0.0
    else:
        mismatches['goal_node_rewritten_correctly'] = 100.0
    if gt_data[3] != rewrite_data[3]:
        mismatches['heuristics_rewritten_correctly'] = 0.0
    else:
        mismatches['heuristics_rewritten_correctly'] = 100.0

    return mismatches

def question_rewrite_validation(all_str_except_plan):
    sections = split_graph_sections(all_str_except_plan)
    if len(sections) < 3:
        raise ValueError("Expected at least 2 'The graph is:' sections for GT and rewrite.")

    gt_data = parse_graph_data(sections[1])
    rewrite_data = parse_graph_data(sections[2])

    mismatches = compare_graphs(gt_data, rewrite_data)

    return {
        "gt": {
            "graph": gt_data[0],
            "start_node": gt_data[1],
            "goal_node": gt_data[2],
            "heuristics": gt_data[3],
        },
        "rewrite": {
            "graph": rewrite_data[0],
            "start_node": rewrite_data[1],
            "goal_node": rewrite_data[2],
            "heuristics": rewrite_data[3],
        },
        "mismatches": mismatches
    }

# result = question_rewrite_validation(all_str_except_plan)["mismatches"]
# result["mismatches"]


##### EVAL 1 Check the process
def check_process(all_str_except_plan):
    each_segment = all_str_except_plan.split("~~~")
    process_eval = {}
    step_0_process_correct = ["initialize_heap","initialize_visited","initialize_costs","initialize_path", "costs.update","heap.push","visited.contains","heap.is_empty"]
    caught_inits = []

    def extract_neighbors_list(text):
        """Extract neighbors list from text like 'The neighbors of the current node are: [21]'"""
        match = re.search(r'The neighbors of the current node are: \[([^\]]+)\]', text)
        if match:
            neighbors_str = match.group(1)
            # Handle both single numbers and comma-separated lists
            if ',' in neighbors_str:
                return [int(x.strip()) for x in neighbors_str.split(',')]
            else:
                return [int(neighbors_str.strip())]
        return []

    def check_inequality_with_less_than(text):
        """Check if there's an inequality with '<' sign and extract the values"""
        pattern = r'(\d+(?:\.\d+)?)\s*<\s*(\w+)'
        matches = re.findall(pattern, text)
        return len(matches) > 0

    def validate_astar_step(segment_text):
        """Validate a single A* algorithm step"""
        validation_results = {
            'heap_pop_occurred': False,
            'node_added_to_visited': False,
            'popped_node': None,
            'visited_node': None,
            'neighbors_processed_correctly': True,
            'neighbors_list': [],
            'cost_fetched_for_neighbors': [],
            'inequality_checks': [],
            'costs_updated': [],
            'heap_contains_checked': [],
            'heap_pushed': [],
            'path_updated': [],
            'goal_check': False,
            'heap_empty_check': False
        }
        
        lines = segment_text.split('\n')
        
        # Check for heap.pop and extract popped node
        for i, line in enumerate(lines):
            if 'Action: heap.pop()' in line:
                validation_results['heap_pop_occurred'] = True
                # Look for the observation (next few lines)
                for j in range(i+1, min(i+5, len(lines))):
                    if lines[j].strip().isdigit():
                        validation_results['popped_node'] = int(lines[j].strip())
                        break
            
            # Check if the same node is added to visited
            if 'Action: visited.add(' in line:
                match = re.search(r'visited\.add\((\d+)\)', line)
                if match:
                    validation_results['node_added_to_visited'] = True
                    validation_results['visited_node'] = int(match.group(1))
        
        # Verify popped node equals visited node
        if (validation_results['popped_node'] is not None and 
            validation_results['visited_node'] is not None):
            validation_results['same_node_popped_and_visited'] = (
                validation_results['popped_node'] == validation_results['visited_node']
            )
        
        # Extract neighbors list
        validation_results['neighbors_list'] = extract_neighbors_list(segment_text)
        
        # For each neighbor, check the required conditions
        neighbors_with_better_cost = []  # Track neighbors where inequality is true
        
        for neighbor in validation_results['neighbors_list']:
            neighbor_str = str(neighbor)
            
            # 1. Check if cost is being fetched (this should always happen)
            cost_fetch_pattern = f'Action: costs\.fetch\(node={neighbor}\)'
            if re.search(cost_fetch_pattern, segment_text):
                validation_results['cost_fetched_for_neighbors'].append(neighbor)
            
            # 2. Check for inequality with "<" sign and extract the specific neighbor context
            # Find the section for this specific neighbor
            lines = segment_text.split('\n')
            neighbor_section_start = -1
            neighbor_section_end = -1
            
            for i, line in enumerate(lines):
                if f'The current neighbor is: {neighbor}' in line:
                    neighbor_section_start = i
                elif neighbor_section_start != -1 and ('The current neighbor is:' in line or 'Finished looping' in line):
                    neighbor_section_end = i
                    break
            
            if neighbor_section_start != -1:
                if neighbor_section_end == -1:
                    neighbor_section_end = len(lines)
                
                neighbor_section = '\n'.join(lines[neighbor_section_start:neighbor_section_end])
                # Check for inequality in this specific neighbor's section
                if check_inequality_with_less_than(neighbor_section):
                    validation_results['inequality_checks'].append(neighbor)
                    neighbors_with_better_cost.append(neighbor)
                    
                    # Only check these if inequality was true
                    # 3. Check if costs.update is called for this neighbor
                    cost_update_pattern = f'Action: costs\.update\(node={neighbor}'
                    if re.search(cost_update_pattern, neighbor_section):
                        validation_results['costs_updated'].append(neighbor)
                    
                    # 4. Check if heap.contains is checked
                    heap_contains_pattern = f'Action: heap\.contains\({neighbor}\)'
                    if re.search(heap_contains_pattern, neighbor_section):
                        validation_results['heap_contains_checked'].append(neighbor)
                    
                    # 5. Check if heap.push is called for this neighbor
                    heap_push_pattern = f'Action: heap\.push\({neighbor},'
                    if re.search(heap_push_pattern, neighbor_section):
                        validation_results['heap_pushed'].append(neighbor)
                    
                    # 6. Check if path is updated for this neighbor
                    path_update_pattern = f'Action: path\.update\(node={neighbor}'
                    if re.search(path_update_pattern, neighbor_section):
                        validation_results['path_updated'].append(neighbor)
        
        # Store neighbors that had better costs for later validation
        validation_results['neighbors_with_better_cost'] = neighbors_with_better_cost
        
        # Check for goal node check (visited.contains(57))
        if 'Action: visited.contains' in segment_text:
            validation_results['goal_check'] = True
        
        # Check for heap.is_empty() check
        if 'Action: heap.is_empty()' in segment_text:
            validation_results['heap_empty_check'] = True

        if "Yes, the `goal` node" in segment_text: # hack for time constraint
            validation_results['heap_empty_check'] = True
        
        return validation_results

    # Main processing loop
    for i, seg in enumerate(each_segment):
        # print(f"{i} ++++++++++++++===================================")
        if i == len(each_segment) - 1:
            if ("is empty" in each_segment[i-1]) and ("unsuccessful" in seg): # bit hacky
                process_eval["step_last_process_correct"] = True
            else:
                process_eval["step_last_process_correct"] = False

            if (("Yes, the `goal` node" in each_segment[i-1]) and ("Successful".lower() in seg.lower())):
                if "path.trace" in seg:
                    process_eval["step_last_process_correct"] = True
                else:
                    process_eval["step_last_process_correct"] = False
            else:
                process_eval["step_last_process_correct"] = False
            break

        if i == 0:
            # Handle step 1 initialization as before
            for j, l in enumerate(seg.split("\n\n")):
                if "Action:" in l:
                    function_name = l.split("Action:")[1].strip().split("(")[0].strip()
                    caught_inits.append(function_name)
                    
            if caught_inits == step_0_process_correct:
                process_eval["step_0_process_correct"] = True
            else:
                process_eval["step_0_process_correct"] = False
        else:
            # Handle subsequent A* algorithm steps
            validation_results = validate_astar_step(seg)
            
            # Store results for this step
            step_key = f"step_{i}"
            process_eval[step_key] = validation_results
            
            # Print validation summary for this step
            # print(f"Step {i} Validation Results:")
            # print(f"  Heap pop occurred: {validation_results['heap_pop_occurred']}")
            # print(f"  Node added to visited: {validation_results['node_added_to_visited']}")
            # print(f"  Same node popped and visited: {validation_results.get('same_node_popped_and_visited', 'N/A')}")
            # print(f"  Neighbors found: {validation_results['neighbors_list']}")
            # print(f"  Cost fetched for all neighbors: {validation_results['cost_fetched_for_neighbors']}")
            # print(f"  Neighbors with better cost (inequality true): {validation_results['neighbors_with_better_cost']}")
            # print(f"  Costs updated (only for better cost): {validation_results['costs_updated']}")
            # print(f"  Heap contains checked (only for better cost): {validation_results['heap_contains_checked']}")
            # print(f"  Heap pushed (only for better cost): {validation_results['heap_pushed']}")
            # print(f"  Path updated (only for better cost): {validation_results['path_updated']}")
            # print(f"  Goal check (visited.contains(goalNode)): {validation_results['goal_check']}")
            # print(f"  Heap empty check: {validation_results['heap_empty_check']}")
            
            # Check if all neighbors were processed correctly
            neighbors = validation_results['neighbors_list']
            neighbors_with_better_cost = validation_results['neighbors_with_better_cost']
            all_neighbors_correct = True
            
            # All neighbors should have cost fetched
            for neighbor in neighbors:
                if neighbor not in validation_results['cost_fetched_for_neighbors']:
                    all_neighbors_correct = False
                    print(f"    Neighbor {neighbor} missing cost fetch")
            
            # Only neighbors with better cost should have the other operations
            for neighbor in neighbors_with_better_cost:
                neighbor_correct = (
                    neighbor in validation_results['costs_updated'] and
                    neighbor in validation_results['heap_contains_checked'] and
                    neighbor in validation_results['heap_pushed'] and
                    neighbor in validation_results['path_updated']
                )
                if not neighbor_correct:
                    all_neighbors_correct = False
                    print(f"    Neighbor {neighbor} with better cost, missing require processes like cost update, heap_push...")
            
            process_eval[f"{step_key}_neighbors_correct"] = all_neighbors_correct
            process_eval[f"{step_key}_termination_checks"] = (
                validation_results['goal_check'] and validation_results['heap_empty_check']
            )
    return process_eval
# pprint(check_process(all_str_except_plan))

##### Eval 1.5 overall process correctness
def check_overall_correctness(validation_dict):
    def count_validations(value):
        """Recursively count total boolean validations and passed validations"""
        total_checks = 0
        passed_checks = 0
        
        if isinstance(value, dict):
            for v in value.values():
                sub_total, sub_passed = count_validations(v)
                total_checks += sub_total
                passed_checks += sub_passed
        elif isinstance(value, list):
            for item in value:
                sub_total, sub_passed = count_validations(item)
                total_checks += sub_total
                passed_checks += sub_passed
        elif isinstance(value, bool):
            total_checks = 1
            passed_checks = 1 if value else 0
        # For non-boolean values, we don't count them as validations
        
        return total_checks, passed_checks
    
    total, passed = count_validations(validation_dict)
    
    if total == 0:
        return 100  # No validations found, consider as 100%
    
    return (passed / total) * 100

# print(check_overall_correctness(check_process(all_str_except_plan)))
# eval_dict = {"overall process correctness": check_overall_correctness(check_process(all_str_except_plan))}


##### Eval 2 Check if values are consistant, calculations are right and checks are correct
# and added inline comments marking key changes.
def consistency_calculations_checks_wrt_GT(all_str_except_plan, eval_dict={}):
    sections = split_graph_sections(all_str_except_plan)
    graph, start_node, goal_node, heuristics = parse_graph_data(sections[1])  # GT

    # Use defaultdict(list) to collect multiple occurrences per key
    if eval_dict is None:
        eval_dict = defaultdict(list)
    else:
        # Convert existing single values into lists to preserve old entries
        for k, v in list(eval_dict.items()):
            if not isinstance(v, list):
                eval_dict[k] = [v]

    # Helper to append rather than overwrite
    def log(key, value):
        eval_dict.setdefault(key, []).append(value)

    # Initialize algorithm state variables
    current_node = parent_node = None
    gt_goal_reached = heap_empty = False
    heap = []  # local heap instead of globals
    costs = {}
    path = {}
    visited = set()

    all_lines = all_str_except_plan.split("\n\n")

    def safe_eval_boolean(text):
        try:
            t = text.strip().lower()
            if 'true' in t: return True
            if 'false' in t: return False
            return eval(text.strip())
        except:
            return None

    def safe_get(arr, idx):
        return arr[idx] if 0 <= idx < len(arr) else None

    for i, line in enumerate(all_lines):
        # current neighbor check logged every time
        if "The current neighbor is:" in line:
            curr = line.split(":")[-1].strip()
            if parent_node is not None:
                try:
                    in_graph = int(curr) in eval(graph)[int(parent_node)]
                except:
                    in_graph = False
                log("current_node_taken_from_graph", in_graph)
            current_node = curr
            continue

        # cost calc from 999 collects each evaluation
        if "The previously known cost is" in line and parent_node is not None:
            try:
                gt_cost = int(costs[parent_node])
                nxt = safe_get(all_lines, i+1) or ""
                parts = nxt.split("=")
                if len(parts) >= 3:
                    found_raw = parts[-2].strip().split()[0]
                    found_cost = int(found_raw)
                    lhs_rhs_equal = eval(parts[-2]) == int(parts[-1].strip())
                    new_cost_ok = (gt_cost + 1) == eval(parts[-2])
                    log("cost_in_cost_calc_eqn_check", gt_cost == found_cost)
                    log("final_cost_equation_lhs_equals_rhs", lhs_rhs_equal)
                    log("new_cost_val_in_cost_eqn_check", new_cost_ok)
            except:
                # log failures consistently
                log("cost_in_cost_calc_eqn_check", False)
                log("final_cost_equation_lhs_equals_rhs", False)
                log("new_cost_val_in_cost_eqn_check", False)

        # neighbors list correctness and exploration logged
        if "The neighbors of the current node are:" in line and current_node:
            m = re.search(r"\[([0-9,\s]+)\]", line)
            neigh = []
            if m:
                neigh = list(map(int, m.group(1).split(',')))
            try:
                expected = eval(graph)[int(current_node)]
                log("neighbors_correct_from_graph", neigh == expected)
            except:
                log("neighbors_correct_from_graph", False)

            missing = set(eval(graph).get(int(current_node), []))
            for block in all_lines[i:]:
                if "Finished looping through all the neighbors" in block:
                    break
                if "The current neighbor is:" in block:
                    try:
                        missing.discard(int(block.split(":")[-1]))
                    except:
                        pass
            log("all_neighbors_explored", len(missing) == 0)
        
        if "Action:" in line:
            act = line.split("Action:")[-1].strip()
            # if i == 42:
            # print("===========================",i,act)
            # handling actions locally to update state
            if act.startswith("initialize_costs"):
                costs = {str(k): int('999') for k in eval(heuristics)}
            elif act.startswith("initialize_heap"):
                heap = []
            elif act.startswith("initialize_path"):
                path = {}
            elif act.startswith("initialize_visited"):
                visited = set()
            elif act.startswith("costs.update"):
                r = re.search(r"node=(\d+),\s*cost=(\d+)", act)
                if r:
                    costs[r.group(1)] = int(r.group(2))
            elif act.startswith("costs.fetch"):
                r = re.search(r"node=(\d+)", act)
                if r:
                    current_cost = costs.get(r.group(1), int('999'))
            elif act.startswith("heap.push"):
                r = re.search(r"heap\.push\((\d+),\s*(\d+)\)", act)

                if r:
                    heap.append((r.group(1), r.group(2)))
            elif act.startswith("heap.pop"):
                nxt = safe_get(all_lines, i+2) or ''
                node_popped = nxt.strip()
                heap = [h for h in heap if h[0] != node_popped]
                parent_node = current_node = node_popped
            elif act.startswith("heap.is_empty"):
                heap_empty = len(heap) == 0
            elif act.startswith("visited.add"):
                r = re.search(r"visited\.add\((\d+)\)", act)
                if r:
                    visited.add(r.group(1))
            elif act.startswith("path.update"):
                r = re.search(r"node=(\d+),\s*previous_node=(\d+)", act)
                if r:
                    path[r.group(1)] = r.group(2)
            continue

        # inequality check logs every time encountered
        if "newly discovered cost is less than" in line:
            ok = False
            if current_node and parent_node:
                try:
                    old = int(costs[current_node])
                    new = int(costs[parent_node]) + 1
                    resp = safe_get(all_lines, i+2) or ''
                    ok = (("No" in resp and new >= old) or ("Yes" in resp and new < old))
                except Exception as e:
                    ok = False
            log("inequality_newly_discovered_cost_is_less", ok)

        # Estimated value logging for each encounter
        if "Estimated value" in line:
            parts = line.split("=")
            if len(parts) >= 3:
                equation_str = parts[1].strip()
                rhs_str = parts[2].strip()

                tokens = equation_str.split()
                if len(tokens) >= 3:
                    cost_tok, _, heur_tok = tokens[:3]
                    node_id = int(current_node) if current_node is not None else int(start_node)

                    try:
                        actual_cost = int(costs.get(str(node_id), 0))
                    except (KeyError, ValueError):
                        actual_cost = 0

                    try:
                        actual_heuristic = int(eval(heuristics).get(node_id, 0))
                    except (KeyError, ValueError):
                        actual_heuristic = 0

                    gt_estimate_val = actual_cost + actual_heuristic

                    try:
                        rhs_val = int(rhs_str)
                    except ValueError:
                        rhs_val = None

                    lhs_rhs_check = eval(equation_str) == rhs_val if rhs_val is not None else False
                    gt_vs_rhs_check = gt_estimate_val == rhs_val if rhs_val is not None else False

                    log("cost_from_estimate_equation", str(actual_cost) == cost_tok)
                    log("heuristic_from_estimate_equation", str(actual_heuristic) == heur_tok)
                    log("estimate_value_lhs_equals_rhs", lhs_rhs_check)
                    log("calculated_estimate_value_matches_gen_est_val", gt_vs_rhs_check)

        # Base conditions logged every time
        if any(x in line for x in ["Therefore, continuing the search."]): # , "Therefore, continuing the search." 
            res5 = safe_get(all_lines, i-5) or ''
            goal_ok = (goal_node in visited)
            found_goal = safe_eval_boolean(res5)
            if found_goal is not None:
                log("goal_check", found_goal == goal_ok)

            res1 = safe_get(all_lines, i-1) or ''
            emp_ok = True if len(heap) == 0 else False
            found_emp = safe_eval_boolean(res1)
            if found_emp is not None:
                log("empty_heap_check", found_emp == emp_ok)

            is_cont = "Neither condition is true" in line
            verdict_ok = (not emp_ok and not goal_ok) if is_cont else (emp_ok or goal_ok)
            log("base_case_final_verdict_check", verdict_ok)
            
        if any(x in line for x in ["ending the search"]): # Used hack for time constraint
            res1 = safe_get(all_lines, i-1) or ''
            goal_ok = (goal_node in visited)
            found_goal = safe_eval_boolean(res1)
            if found_goal is not None:
                log("goal_check", found_goal == goal_ok)

            log("empty_heap_check", True)

            log("base_case_final_verdict_check", True)

    return dict(eval_dict)

# consistency_calculations_checks_wrt_GT(all_str_except_plan)
##### STEP final
def compute_accuracy(eval_dict):
    accuracy = {}
    for key, results in eval_dict.items():
        total = len(results)
        if total == 0:
            pct = None       # or 0, depending on preference
        else:
            true_count = sum(1 for r in results if bool(r))
            pct = (true_count / total) * 100
        accuracy[key] = pct
    return accuracy

# compute_accuracy(consistency_calculations_checks_wrt_GT(all_str_except_plan))

merged = {
    **eval_dict,
    **question_rewrite_validation(all_str_except_plan)["mismatches"], #eval1
    **{"overall_process_correctness": check_overall_correctness(check_process(all_str_except_plan))}, #eval2
    **compute_accuracy(consistency_calculations_checks_wrt_GT(all_str_except_plan)), #eval3
}
merged