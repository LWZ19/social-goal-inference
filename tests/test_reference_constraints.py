"""The solver with reference constraints: a test of the solver,
not a measure of a run.

For every generated scene of the four situations named in COVERED, the constraints a careful answer would hold are
written from who is in the group at the target; for a scene of another situation none is given. The people are those of
ground truth as the method's `oracle` source gives them, and the solver's own properties are asserted: the place breaks
no exclusion where some cell breaks none, no selectable cell comes before it in the solver's order,
and no selectable cell lies outside the scene's static free space. Where people watch the target it is measured as
well, from the place itself, that the robot stands level with the hindmost of them or behind.

What the scorer says of those places is printed, per situation, as a number with no bar to meet: no success rate is
asserted here, and nothing in the solver was fitted to one. `pytest -rP` shows it, and so does `pytest -s`.
"""
import math

from sgi import run as harness
from sgi import solver
from sgi.config import load_config
from sgi.evaluate import score
from sgi.freespace import static_grid
from sgi.gt.derive import oracle_target, prepare
from sgi.scene import load_scene
from sgi.stages.goal import standard_goal
from sgi.stages.people import number_people
from sgi.types import GoalResult
from tests.solver_layouts import Layout, ask
from tests.test_run import placeholder, scene_dirs  # noqa: F401
from tests.test_solver import ahead_of, in_order

SPLITS = ("eval", "examples")
# The situations reference constraints are written for below. The placeholder set also has scenes of a picture being
# taken and of people waiting at a lift, the two held-out situations: nobody has said what a careful answer holds
# there, and this test invents none. In those scenes the solver is given no constraint at all, and what is asserted
# of its place is what holds whatever it is given.
COVERED = ("no_one", "waiting_line", "conversation_group", "audience")
MM = 1e-3                                          # the solver compares its values rounded to the millimetre


def reference_constraints(situation, members, needed):
    """The constraints of a careful answer, from who is in the group at the target: `members` are their numbers,
    `needed` the number of the person the robot was sent to, None when it was sent to a thing."""
    # This is the one place where the tests look at a scene's situation to decide what the method is given, and it is
    # the harness doing so, to choose reference constraints. The solver is given the constraints and never the label.
    if situation == "waiting_line":                # the members wait their turn at the target
        return [ask("come_after", *members)]
    if situation == "conversation_group":          # the members attend to one another, and one of them is needed
        return [ask("keep_out_of_shared_space", *members), ask("visible_to", needed), ask("face", needed)]
    if situation == "audience":                    # the members watch the target, and the robot is to look at it too
        return [ask("keep_out_of_view", *members, target=True), ask("face", target=True)]
    assert situation == "no_one", situation        # nobody is near: the solver runs on what it applies by itself
    return []


def numbered_people(truth, observation, target, th):
    """The people of ground truth as the method has them under `people: oracle`: those in view, merged, cut and
    numbered. With them, the number of each by their id in ground truth."""
    people, cut = number_people(harness.OraclePeople(truth, th)(observation).found, target, observation, th, th.max_people)
    number = {person.id: found.number for found in people for person in truth.people if (person.x, person.y) == found.xy}
    assert not cut and len(number) == len(people)
    return people, number


def test_with_reference_constraints_the_place_has_the_solvers_own_properties_on_every_placeholder_scene(placeholder):
    cfg = load_config()
    th = cfg.thresholds
    tally = {}                                     # by split and situation: the episodes, and those the scorer calls a success
    for split in SPLITS:
        for folder in scene_dirs(placeholder / split):
            inputs, truth = load_scene(folder)
            observation, truths = inputs.observation, prepare(inputs, truth, cfg)
            static = static_grid(truth, th.grid_resolution_m)
            for episode in inputs.episodes:
                where = (inputs.scene_id, episode.episode_id)
                gt = truths[episode.episode_id].gt
                wanted = truth.targets[gt.target_ids[0]]
                target = oracle_target(observation, wanted, th)       # as oracle grounding gives it
                people, number = numbered_people(truth, observation, target, th)
                group = truth.group_for(episode.episode_id, wanted.id)
                members = [] if group is None else [number[person.id] for person in truth.group(group) if person.id in number]
                answer = reference_constraints(gt.situation, members, number.get(wanted.person_id)) if gt.situation in COVERED else []
                lay = Layout(observation, people, target, standard_goal(observation, target, th))
                # The place is a selectable cell, no selectable cell comes before it in the order, and it breaks no
                # exclusion where some cell breaks none.
                solved = in_order(lay, *answer, th=th)
                assert [use.status for use in solved.uses] == ["used"] * len(answer), where
                if gt.situation == "audience":                        # everyone who watches is before the thing: all are joined,
                    assert solved.note.not_ahead_of == sorted(members) and len(members) >= 3, where
                    assert not solved.flags.exclusion_broken, where   # and floor level with the hindmost of them is in view
                    # That the view is not broken is the solver's own figure, the `broken_m` it records, like the flag
                    # above: neither is measured here.
                    assert solved.uses[0].broken_m == 0.0, where
                    # Measured here from the place itself, with no word of the solver's taken for it and no scorer:
                    # along the direction from the members' middle to the target the place is no farther on than the
                    # hindmost of them, to the millimetre the solver compares in.
                    watching = [found.xy for found in people if found.number in members]
                    assert ahead_of(solved.pose_world, watching, target.world) <= MM, where
                if not solved.flags.exclusion_broken:                 # measured here too: it is inside nobody's own space
                    assert all(math.dist(solved.pose_world[:2], person.xy) > th.standoff_m - MM for person in people), where
                floor = solver.floor(observation, people, th)
                outside = [xy for xy in floor.centres()[floor.selectable] if not static.is_free(*xy)]
                assert outside == [], where                           # no place lies outside the static free space
                result = GoalResult(status="ok", source="constraints", pose_world=solved.pose_world,
                                    pose_robot=solved.pose_robot, target=target)
                count = tally.setdefault((split, gt.situation), [0, 0])
                count[0] += 1
                count[1] += score(result, observation, truth, gt, th).success
    given = sum(episodes for (_, situation), (episodes, _) in tally.items() if situation in COVERED)
    alone = sum(episodes for (_, situation), (episodes, _) in tally.items() if situation not in COVERED)
    assert (given, alone) == (52, 8)                   # two episodes a scene, as the set is generated now: 26 scenes solved with
    #                                                    reference constraints, and 4 of the two held-out situations with none
    assert {situation for _, situation in tally} == {*COVERED, "group_photo", "lift_wait"}
    assert sum(episodes for (_, situation), (episodes, _) in tally.items() if situation == "audience") == 6
    print("the solver with reference constraints: the scorer's success per situation, with no bar to meet")
    for (split, situation), (episodes, successes) in sorted(tally.items(), key=lambda item: (SPLITS.index(item[0][0]), item[0][1])):
        print(f"  {split:9s}{situation:19s}{successes:3d} of {episodes}{'' if situation in COVERED else '   (no constraint given)'}")
