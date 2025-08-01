from py_arg.abstract_argumentation_classes.abstract_argumentation_framework import AbstractArgumentationFramework
from py_arg.abstract_argumentation_classes.argument import Argument
from py_arg.abstract_argumentation_classes.defeat import Defeat
from typing import Callable, Tuple, List
import random

###--------------------------- AF GENERATORS ---------------------------###

def generate_no_conflict(n: int) -> AbstractArgumentationFramework:
    """Generates a No-Conflict (NC) AF with n arguments and no attacks. (AF TYPE 1)"""
    if n <= 0: raise ValueError("Number of arguments 'n' must be positive.")
    return AbstractArgumentationFramework(f'No-Conflict (n={n})', [Argument(f"A{i+1}") for i in range(n)], [])

def generate_linear_attack_chain(n: int) -> AbstractArgumentationFramework:
    """Generates a Linear Attack Chain (LAC) AF with n arguments. (AF TYPE 2)"""
    if n < 2: raise ValueError("Linear Attack Chain requires at least 2 arguments.")
    args = [Argument(f"A{i+1}") for i in range(n)]
    defs = [Defeat(args[i], args[i+1]) for i in range(n - 1)]
    return AbstractArgumentationFramework(f'Linear-Attack-Chain (n={n})', args, defs)

def generate_cycle(n: int) -> AbstractArgumentationFramework:
    """Generates a Cycle (CYC) AF with n arguments. (AF TYPE 3)"""
    if n < 2: raise ValueError("Cycle requires at least 2 arguments.")
    args = [Argument(f"A{i+1}") for i in range(n)]
    defs = [Defeat(args[i], args[(i + 1) % n]) for i in range(n)]
    return AbstractArgumentationFramework(f'Cycle (n={n})', args, defs)

def generate_single_target_multiple_attackers(n: int) -> AbstractArgumentationFramework:
    """Generates a Single Target, Multiple Attackers (STMA) AF. (AF TYPE 4)"""
    if n < 2: raise ValueError("This structure requires at least 1 attacker (n>=2).")
    target = Argument("T")
    attackers = [Argument(f"A{i+1}") for i in range(n - 1)]
    defs = [Defeat(attacker, target) for attacker in attackers]
    return AbstractArgumentationFramework(f'Single-Target-Multiple-Attackers (n={n})', [target] + attackers, defs)

def generate_single_attack_multiple_defenders(n: int) -> AbstractArgumentationFramework:
    """Generates a Single Attack, Multiple Defenders (SAMD) AF. (AF TYPE 5)"""
    if n < 3: raise ValueError("This structure requires at least 1 defender (n>=3).")
    target, attacker = Argument("T"), Argument("Att")
    defenders = [Argument(f"D{i+1}") for i in range(n - 2)]
    defs = [Defeat(attacker, target)] + [Defeat(defender, attacker) for defender in defenders]
    return AbstractArgumentationFramework(f'Single-Attack-Multiple-Defenders (n={n})', [target, attacker] + defenders, defs)

def generate_disconnected_symmetric_pairs(n: int) -> AbstractArgumentationFramework:
    """Generates a Disconnected Symmetric Pairs (DSP) AF. (AF TYPE 6)"""
    if n < 2 or n % 2 != 0: raise ValueError("Number of arguments 'n' must be an even number >= 2.")
    num_pairs = n // 2
    args, defs = [], []
    for i in range(num_pairs):
        arg1, arg2 = Argument(f"A{i+1}"), Argument(f"B{i+1}")
        args.extend([arg1, arg2])
        defs.extend([Defeat(arg1, arg2), Defeat(arg2, arg1)])
    return AbstractArgumentationFramework(f'Disconnected-Symmetric-Pairs (n={n})', args, defs)

###--------------------------- METAMORPHIC TRANSFORMATIONS ---------------------------###

def apply_isomorphism(af: AbstractArgumentationFramework, renaming_map: Callable[[str], str] = lambda name: f"X_{name}") -> Tuple[AbstractArgumentationFramework, Callable[[str], str]]:
    """Applies an isomorphic transformation by renaming all arguments."""
    old_to_new_map = {old_arg.name: Argument(renaming_map(old_arg.name)) for old_arg in af.arguments}
    new_args = list(old_to_new_map.values())
    new_defs = [Defeat(old_to_new_map[old_d.from_argument.name], old_to_new_map[old_d.to_argument.name]) for old_d in af.defeats]
    new_af = AbstractArgumentationFramework(f"isomorphic_{af.name}", new_args, new_defs)
    return new_af, renaming_map

def apply_fundamental_consistency(af: AbstractArgumentationFramework, sa_name: str = "SA") -> Tuple[AbstractArgumentationFramework, str]:
    """Adds a new, isolated, self-defeating argument to the framework."""
    sa_argument = Argument(sa_name)
    new_args = af.arguments + [sa_argument]
    new_defs = af.defeats + [Defeat(sa_argument, sa_argument)]
    new_af = AbstractArgumentationFramework(f"fundamental_consistency_{af.name}", new_args, new_defs)
    return new_af, sa_name

def apply_modularity(af: AbstractArgumentationFramework, u_name: str = "U") -> Tuple[AbstractArgumentationFramework, str]:
    """Adds a new, isolated argument with no interactions."""
    u_argument = Argument(u_name)
    new_args = af.arguments + [u_argument]
    new_af = AbstractArgumentationFramework(f"modularity_{af.name}", new_args, af.defeats)
    return new_af, u_name

def apply_defense_dynamics(af: AbstractArgumentationFramework) -> Tuple[AbstractArgumentationFramework, Argument, Argument, Argument]:
    """Adds a defender to a randomly selected attack relationship."""
    if not af.defeats:
        raise ValueError("Cannot apply Defense Dynamics to an AF with no defeats.")
    random_defeat = random.choice(af.defeats)
    attacker, target = random_defeat.from_argument, random_defeat.to_argument
    defender_name = "M_Defender"
    counter = 0
    while af.is_in_arguments(defender_name) or any(defender_name == arg.name for arg in [attacker, target]):
        counter += 1
        defender_name = f"M_Defender_{counter}"
    defender = Argument(defender_name)
    new_args = af.arguments + [defender]
    new_defs = af.defeats + [Defeat(defender, attacker)]
    new_af = AbstractArgumentationFramework(f"defense_dynamics_{af.name}", new_args, new_defs)
    return new_af, defender, attacker, target
