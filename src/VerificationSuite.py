from typing import Dict, List, Callable, Tuple, Set, FrozenSet

# Third-party imports
from dotenv import load_dotenv

from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from py_arg.abstract_argumentation_classes.argument import Argument
from py_arg.algorithms.semantics.is_conflict_free import is_conflict_free
from py_arg.algorithms.semantics.get_complete_extensions import get_complete_extensions
from py_arg.algorithms.semantics.get_grounded_extension import get_grounded_extension
from py_arg.algorithms.semantics.get_preferred_extensions import get_preferred_extensions
from py_arg.algorithms.semantics.get_stable_extensions import get_stable_extensions

# Load environment variables from .env file
load_dotenv()

class VerificationSuite:
    """
    Isolates all verification logic. This class is now responsible for
    verifying both Fundamental Properties (FPs) of a single AF's extensions
    and the Metamorphic Relations (MRs) between two sets of extensions
    (from an original AF and a transformed AF).
    """
    def __init__(self):
        """Initialize the verification suite with required dependencies."""
        self.is_conflict_free = is_conflict_free
        print("Initialized Verification Suite.")

    def _str_to_args(self, arg_names: List[str], af: AbstractArgumentationFramework) -> List[Argument]:
        """Helper to convert a list of argument names back to Argument objects. 
        --> used when converting the LLM's list of argument names to Argument objects.
        
        Args:
            arg_names (List[str]): List of argument names as strings.
            af (AbstractArgumentationFramework): The argumentation framework containing the arguments.
            
        Returns:
            List[Argument]: List of Argument objects corresponding to the given names.
            
        Raises:
            ValueError: If any argument name is not found in the argumentation framework.
        """
        return [af.get_argument(name) for name in arg_names]

    def _sets_of_str_to_str(self, sets: List[List[str]]) -> str:
        """Helper to format sets of argument names for readable error messages.
        --> sorts the argument names within each set in alphabetical order for consistency
        
        Args:
            sets (List[List[str]]): List of sets of argument names.
        Returns:
            str: A string representation of the sets with each set's elements sorted.
        """
        return str([sorted(s) for s in sets])
    
    def verify_output_schema(self, llm_extensions: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
        violations = {}
        # Reminder: LLM computed extensions is of Type List[List[str]]
        for ext_type in ['GE', 'CE', 'PE', 'SE']:
            ext_sets = llm_extensions.get(ext_type)
            
            # Checks if each set of extensions is a list of strings.
            try:
                # Check if LLM's answer is a mathematical notation, or text description; which is not acceptable.
                if any(" " in a for s in ext_sets for a in s):
                    msg = f'Error parsing LLM extensions: {ext_sets}. LLM used mathematical notations or text description of the extensions instead of argument names.'
                    violations.setdefault("SCHEMA-ERROR", []).append(msg)
                    return violations
                
                if any(not isinstance(a, str) for s in ext_sets for a in s):
                    msg = f'Error parsing LLM extensions: {ext_sets}. LLM answers contain members that is not a string.'
                    violations.setdefault("SCHEMA-ERROR", []).append(msg)
                    return violations
                
            except Exception as e:
                msg = f"Cannot verify the schema for the computed extensions ({ext_sets}): {e}"
                violations.setdefault("SCHEMA-VERIFICATION-FAILED", []).append(msg)
                return violations
            
    def verify_output_schema_metamorphic(self, original_ext: Dict[str, List[List[str]]], transformed_ext: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
        violations = {}
        # Reminder: LLM computed extensions is of Type List[List[str]]
        for ext_type in ['GE', 'CE', 'PE', 'SE']:
            ext_sets = original_ext.get(ext_type)
            trans_ext_sets = transformed_ext.get(ext_type)
                
            # Checks if all sets of extensions are lists of strings.
            try:
                # Check if LLM's answer is a mathematical notation, or text description; which is not acceptable.
                if any(" " in a for s in ext_sets for a in s) or any(" " in a for s in trans_ext_sets for a in s):
                    msg = f'Error parsing LLM extensions: {ext_sets} or {trans_ext_sets}. LLM used mathematical notations or text description of the extensions instead of argument names.'
                    violations.setdefault("SCHEMA-ERROR", []).append(msg)
                    return violations
                
                invalid_schema_found_in_ext_sets = any(not isinstance(a, str) for s in ext_sets for a in s)
                invalid_schema_found_in_trans_ext_sets = any(not isinstance(a, str) for s in trans_ext_sets for a in s)
                found_invalid_schema = invalid_schema_found_in_ext_sets or invalid_schema_found_in_trans_ext_sets
            
            except Exception as e:
                msg = f"Cannot verify the schema for the computed extensions ({ext_sets} and {trans_ext_sets}): {e}"
                violations.setdefault("SCHEMA-VERIFICATION-FAILED", []).append(msg)
                return violations
            
            if invalid_schema_found_in_ext_sets:
                msg = f'Error parsing LLM extensions: {ext_sets}. LLM answers contain members that is not a string.'
                violations.setdefault("SCHEMA-ERROR", []).append(msg)
            if invalid_schema_found_in_trans_ext_sets:
                msg = f'Error parsing LLM extensions: {trans_ext_sets}. LLM answers contain members that is not a string.'
                violations.setdefault("SCHEMA-ERROR", []).append(msg)
            
            if found_invalid_schema:
                return violations

    def verify_fundamental_properties(self, af: AbstractArgumentationFramework, llm_extensions: Dict[str, List[List[str]]]) -> Dict[str, List[str]]:
        """
        Verifies the fundamental properties of a set of extensions for a single AF.
        
        Checks six fundamental properties (FP-1 to FP-6) that should hold for any valid set of extensions:
        - FP-1: Grounded Extension is a Complete Extension
        - FP-2: Preferred Extension is a Complete Extension
        - FP-3: Stable Extension implies Preferred Extension
        - FP-4: Uniqueness of the Grounded Extension
        - FP-5: Existence of Preferred Extension(s)
        - FP-6: Conflict-Freeness of all extensions
        
        Args:
            af (AbstractArgumentationFramework): The argumentation framework being tested.
            llm_extensions (Dict[str, List[List[str]]]): Extensions generated by the LLM, organized by type.
            
        Returns:
            Dict[str, List[str]]: Dictionary mapping violation types to lists of violation messages.
        """
        
        violations = {}
        
        # Reminder: LLM computed extensions is of Type List[List[str]]
        schema_violations = self.verify_output_schema(llm_extensions)
        if schema_violations:
            return schema_violations
        
        # Converts the sets of argument names to frozensets so we can perform set comparisons.
        ge_sets = {frozenset(s) for s in llm_extensions.get('GE', [])}
        ce_sets = {frozenset(s) for s in llm_extensions.get('CE', [])}
        pe_sets = {frozenset(s) for s in llm_extensions.get('PE', [])}
        se_sets = {frozenset(s) for s in llm_extensions.get('SE', [])}

        # FP-1: Grounded Extension is a Complete Extension
        if not ge_sets.issubset(ce_sets):
            msg = f"FP-1 Violation: Grounded extensions {self._sets_of_str_to_str(list(ge_sets))} are not a subset of Complete extensions {self._sets_of_str_to_str(list(ce_sets))}"
            violations.setdefault("FP-1", []).append(msg)

        # FP-2: Preferred Extension is a Complete Extension
        if not pe_sets.issubset(ce_sets):
            msg = f"FP-2 Violation: Preferred extensions {self._sets_of_str_to_str(list(pe_sets))} are not a subset of Complete extensions {self._sets_of_str_to_str(list(ce_sets))}"
            violations.setdefault("FP-2", []).append(msg)

        # FP-3: Stable Extension implies Preferred Extension
        if not se_sets.issubset(pe_sets):
            msg = f"FP-3 Violation: Stable extensions {self._sets_of_str_to_str(list(se_sets))} are not a subset of Preferred extensions {self._sets_of_str_to_str(list(pe_sets))}"
            violations.setdefault("FP-3", []).append(msg)

        # FP-4: Uniqueness of the Grounded Extension
        if len(ge_sets) > 1:
            msg = f"FP-4 Violation: Expected 1 Grounded extension, but found {len(ge_sets)}: {self._sets_of_str_to_str(list(ge_sets))}"
            violations.setdefault("FP-4", []).append(msg)

        # FP-5: Existence of Preferred Extension(s)
        if len(pe_sets) == 0:
            msg = f"FP-5 Violation: Expected at least 1 Preferred extension, but found none."
            violations.setdefault("FP-5", []).append(msg)

        # FP-6: Conflict-Freeness
        for ext_type, ext_sets in llm_extensions.items():
            for arg_set_str in ext_sets:
                try:
                    # Converts string arguments to an Argument object to verify conflict-freeness.
                    arg_set = self._str_to_args(arg_set_str, af)
                    if not self.is_conflict_free(arg_set, af):
                        msg = f"FP-6 Violation ({ext_type}): Extension {sorted(arg_set_str)} is not conflict-free."
                        violations.setdefault("FP-6", []).append(msg)
                except ValueError as e:
                    msg = f"FP-6 setup error: Could not verify conflict-freeness for {sorted(arg_set_str)}. Reason: {e}"
                    violations.setdefault("FP-6", []).append(msg)

        return violations

    def verify_validity(self, af: AbstractArgumentationFramework, llm_extensions: Dict[str, List[List[str]]]) -> Tuple[Dict[str, List[str]], Dict[str, Set[FrozenSet[Argument]]]]:
        """Verify the validity of the LLM computed extensions using the get_extensions module imported from PyArg"""
        violations = {}
        all_expected_ext_sets = {}
        
        extension_calculator = {
            'GE': get_grounded_extension,
            'CE': get_complete_extensions,
            'PE': get_preferred_extensions,
            'SE': get_stable_extensions
        }
        
        # Reminder: LLM computed extensions is of Type List[List[str]]
        schema_violations = self.verify_output_schema(llm_extensions)
        if schema_violations:
            return schema_violations, {}
        
        for ext_type in ['GE', 'CE', 'PE', 'SE']:
            # Gets the LLM computed extensions for the current extension type.
            ext_sets = llm_extensions.get(ext_type, [])
            
            # Computes the expected extensions using the get_extensions module.
            # The function get_extensions returns Set[FrozenSet[Argument]]
            expected_ext_sets = extension_calculator[ext_type](af)

            if ext_type == 'GE':
                # For get_grounded_extension, the function returns Set[Argument] -> Converts it to Set[FrozenSet[Argument]]
                expected_ext_sets = {frozenset(expected_ext_sets)}
            
            # Converts LLM computed extension sets to Set[FrozenSet[Argument]]
            # 1. Converts ext_sets into List[List[Argument]] (List of Argument Objects instead of strings)
            ext_sets = [self._str_to_args(s, af) for s in ext_sets]
            
            # 2. Converts ext_sets into Set[FrozenSet[Argument]]
            ext_sets = {frozenset(s) for s in ext_sets}
            
            # Compares ext_sets with expected_ext_sets
            if ext_sets != expected_ext_sets:
                msg = (f"Expected {self._sets_of_str_to_str(list(expected_ext_sets))} "
                       f"but got {self._sets_of_str_to_str(list(ext_sets))}")
                violations.setdefault(f'Invalid {ext_type}', []).append(msg)
            else:
                pass
            
            # Converts frozen set back into a list to maintain consistency with the input format. (use [] not {})
            converted_expected_ext_sets = [list(s) for s in expected_ext_sets]
            all_expected_ext_sets[ext_type] = converted_expected_ext_sets
            
        return violations, all_expected_ext_sets
        
        
    
    def verify_isomorphism(self, original_ext: Dict[str, List[List[str]]], transformed_ext: Dict[str, List[List[str]]], renaming_map: Callable[[str], str]) -> Dict[str, List[str]]:
        """Verify the isomorphism metamorphic relation between original and transformed extensions.
        
        Args:
            original_ext (Dict[str, List[List[str]]]): Extensions from the original AF.
            transformed_ext (Dict[str, List[List[str]]]): Extensions from the transformed AF.
            renaming_map (Callable[[str], str]): Function that maps original argument names to renamed ones.
        Returns:
            Dict[str, List[str]]: Dictionary of violations, if any.
        """
                
        violations = {}
        
        # Reminder: LLM computed extensions is of Type List[List[str]]
        schema_violations = self.verify_output_schema_metamorphic(original_ext, transformed_ext)
        if schema_violations:
            return schema_violations
        
        
        for ext_type in ['GE', 'CE', 'PE', 'SE']:
            original_sets = {frozenset(s) for s in original_ext.get(ext_type)}
            transformed_sets = {frozenset(s) for s in transformed_ext.get(ext_type)}
            
            # For each member in a single answer set, use the rename the argument according to the renaming map.
            expected_transformed_sets = {frozenset(renaming_map(arg) for arg in s) for s in original_sets}
            if expected_transformed_sets != transformed_sets:
                msg = (f"MR-ISO Violation ({ext_type}): "
                       f"Expected {self._sets_of_str_to_str(list(expected_transformed_sets))} "
                       f"but got {self._sets_of_str_to_str(list(transformed_sets))}")
                violations.setdefault("MR-ISO", []).append(msg)
        return violations

    def verify_fundamental_consistency(self, original_ext: Dict[str, List[List[str]]], transformed_ext: Dict[str, List[List[str]]], sa_name: str) -> Dict[str, List[str]]:
        """Verify the fundamental consistency metamorphic relation.
        
        Checks that a self-attacking argument is not in any extension and that the grounded extension remains unchanged.
        
        Args:
            original_ext (Dict[str, List[List[str]]]): Extensions from the original AF.
            transformed_ext (Dict[str, List[List[str]]]): Extensions from the transformed AF with a self-attacking argument.
            sa_name (str): Name of the self-attacking argument added to the AF.
        Returns:
            Dict[str, List[str]]: Dictionary of violations, if any.
        """
        violations = {}
        
        # Reminder: LLM computed extensions is of Type List[List[str]]
        schema_violations = self.verify_output_schema_metamorphic(original_ext, transformed_ext)
        if schema_violations:
            return schema_violations
        
        for ext_type, ext_sets in transformed_ext.items():
            for s in ext_sets:
                # MR-FC.1 SA cannot appear in any extension.
                if sa_name in s:
                    msg = (f"MR-FC.1 Violation ({ext_type}): Self-attacking argument '{sa_name}' appeared in extension {sorted(s)}.")
                    violations.setdefault("MR-FC", []).append(msg)
              
        # MR-FC.2 GE must remain unchanged.
        original_ge = {frozenset(s) for s in original_ext.get('GE')}
        transformed_ge = {frozenset(s) for s in transformed_ext.get('GE')}
        
        if original_ge != transformed_ge:
            msg = (f"MR-FC.2 Violation (GE): Original GE {self._sets_of_str_to_str(list(original_ge))} "
                   f"is not equal to transformed GE {self._sets_of_str_to_str(list(transformed_ge))}.")
            violations.setdefault("MR-FC", []).append(msg)
        return violations

    def verify_modularity(self, original_ext: Dict[str, List[List[str]]], transformed_ext: Dict[str, List[List[str]]], u_name: str) -> Dict[str, List[str]]:
        """Verify the modularity metamorphic relation.
        
        Checks that adding an isolated argument results in the expected extensions.
        
        Args:
            original_ext (Dict[str, List[List[str]]]): Extensions from the original AF.
            transformed_ext (Dict[str, List[List[str]]]): Extensions from the transformed AF with an isolated argument.
            u_name (str): Name of the isolated argument added to the AF.
        Returns:
            Dict[str, List[str]]: Dictionary of violations, if any.
        """
        violations = {}
        
        # Reminder: LLM computed extensions is of Type List[List[str]]
        schema_violations = self.verify_output_schema_metamorphic(original_ext, transformed_ext)
        if schema_violations:
            return schema_violations
        
        # --- Verify GE (special case based on set union: GE_new = GE_old U {u}) ---
        original_ge_str = original_ext.get('GE', [])
        transformed_ge_str = transformed_ext.get('GE', [])

        # The GE is unique. We robustly get the list of args, handling both `[]` (no GE found) and `[[]]` (empty set GE).
        # Just a clever way to retrieve the first set of LLM's calculated GE without writing additonal if statements.
        original_ge_list = next(iter(original_ge_str), [])
        transformed_ge_list = next(iter(transformed_ge_str), [])

        # Calculate the expected GE by performing the set union.
        expected_ge_set = frozenset(original_ge_list + [u_name])
        
        if frozenset(transformed_ge_list) != expected_ge_set:
            msg = f"MR-MOD Violation (GE): Expected {sorted(list(expected_ge_set))} but got {sorted(transformed_ge_list)}"
            violations.setdefault("MR-MOD", []).append(msg)

        # --- Verify CE, PE, SE (based on bijection: h(E) = E U {u}) ---
        for ext_type in ['CE', 'PE', 'SE']:
            original_sets_str = original_ext.get(ext_type, [])
            transformed_sets_str = transformed_ext.get(ext_type, [])

            original_sets = {frozenset(s) for s in original_sets_str}
            transformed_sets = {frozenset(s) for s in transformed_sets_str}
            
            # The expected set is a bijection. For each original extension, add 'u_name'.
            expected_transformed_sets = {frozenset(list(s) + [u_name]) for s in original_sets}
            
            if expected_transformed_sets != transformed_sets:
                msg = (f"MR-MOD Violation ({ext_type}): Expected {self._sets_of_str_to_str(list(expected_transformed_sets))} "
                       f"but got {self._sets_of_str_to_str(list(transformed_sets))}")
                violations.setdefault("MR-MOD", []).append(msg)
                
        return violations

    def verify_defense_dynamics(self, original_ext: Dict[str, List[List[str]]], transformed_ext: Dict[str, List[List[str]]], defender: Argument, attacker: Argument, target: Argument) -> Dict[str, List[str]]:
        """Verify the defense dynamics metamorphic relation.
        
        Checks that adding a defender for a target argument has the expected effects on extensions.
        
        Args:
            original_ext (Dict[str, List[List[str]]]): Extensions from the original AF.
            transformed_ext (Dict[str, List[List[str]]]): Extensions from the transformed AF with a defender.
            defender (Argument): The defender argument added to the AF.
            attacker (Argument): The attacker argument that the defender attacks.
            target (Argument): The target argument that is defended.
        Returns:
            Dict[str, List[str]]: Dictionary of violations, if any.
        """
        violations = {}
        
        # Reminder: LLM computed extensions is of Type List[List[str]]
        schema_violations = self.verify_output_schema_metamorphic(original_ext, transformed_ext)
        if schema_violations:
            return schema_violations
        
        original_ge_set = frozenset(next(iter(original_ext.get('GE', [[]])), []))
        transformed_ge_set = frozenset(next(iter(transformed_ext.get('GE', [[]])), []))
        
        # MR-DD.1: The new defender must be in the new Grounded Extension.
        if defender.name not in transformed_ge_set:
            violations.setdefault("MR-DD.1", []).append(f"Defender '{defender.name}' not in new GE.")
        
        # MR-DD.2: The attacker must NOT be in the new Grounded Extension.
        if attacker.name in transformed_ge_set:
            violations.setdefault("MR-DD.2", []).append(f"Attacker '{attacker.name}' still in new GE.")
        
        # MR-DD.3: The target must be reinstated if and only if all of its attackers that were originally "in" are now "out".
        premise_target_was_out = target.name not in original_ge_set
        
        if premise_target_was_out:
            # Get all attackers of the target from the original framework structure.
            all_target_attackers = target.get_ingoing_defeat_arguments
            
            # Condition for reinstatement: Are ALL of the target's attackers now defeated by the new GE?
            # This is true if none of the target's attackers are IN the new GE.
            should_be_reinstated = all(
                ind_attacker.name not in transformed_ge_set 
                for ind_attacker in all_target_attackers
            )

            is_reinstated = target.name in transformed_ge_set

            if should_be_reinstated and not is_reinstated:
                msg = (f"Target '{target.name}' was not reinstated into new GE when all its attackers were defeated.")
                violations.setdefault("MR-DD.3", []).append(msg)
            
            if not should_be_reinstated and is_reinstated:
                # This catches cases where T is reinstated even though it's still attacked by another "in" argument.
                still_in_attackers = [a.name for a in all_target_attackers if a.name in transformed_ge_set]
                msg = (f"Target '{target.name}' was reinstated but should not have been (it is still attacked by {still_in_attackers}).")
                violations.setdefault("MR-DD.3", []).append(msg)

        return violations