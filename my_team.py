# baseline_team.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# baseline_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util
from bokeh.layouts import layout
from boto.gs.lifecycle import LEGAL_ACTIONS
from boto.utils import get_aws_metadata
import random
from capture_agents import CaptureAgent
from dask.config import paths
from game import Directions
from game import Actions
from pyexpat import features
from sqlalchemy.sql.functions import next_value
from sympy.physics.units import energy
from util import nearest_point

from contest.capture import SCARED_TIME
from contest.distance_calculator import get_distance_on_grid
from contest.layout import get_layout
from util import Queue


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None
        self.scared_time = 0
        self.our_scared_time = 0
        self.def_scared_time = 0
        self.max_food = 30
        self.food_left = 0
        self.carried_food = 0
        self.patrolling = set()


    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        print("im in the reflexagent")

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        # if food_left <= 2:
        #     best_dist = 9999
        #     best_action = None
        #     for action in actions:
        #         successor = self.get_successor(game_state, action)
        #         pos2 = successor.get_agent_position(self.index)
        #         dist = self.get_maze_distance(self.start, pos2)
        #         if dist < best_dist:
        #             best_action = action
        #             best_dist = dist
        #     return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}

class BFSSearchProblem:
    def __init__(self, gameState, start, goal):
        self.start = start
        self.goal = goal
        self.walls = gameState.get_walls()

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state == self.goal

    def getSuccessors(self, state):
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            DIRECTION_VECTORS = {
                Directions.NORTH: (0, 1),
                Directions.SOUTH: (0, -1),
                Directions.EAST: (1, 0),
                Directions.WEST: (-1, 0),
            }
            dx, dy = DIRECTION_VECTORS[direction]

            nextState = (int(x + dx), int(y + dy))

            if not self.walls[nextState[0]][nextState[1]]:  # Only allow valid moves
                successors.append((nextState, direction, 1))  # Cost = 1 per step

        return successors


class CapsuleSearchProblem:
    def __init__(self, gameState, start, goal):
        self.start = start
        self.goal = goal
        self.walls = gameState.get_walls()

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state == self.goal

    def getSuccessors(self, state):
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            DIRECTION_VECTORS = {
                Directions.NORTH: (0, 1),
                Directions.SOUTH: (0, -1),
                Directions.EAST: (1, 0),
                Directions.WEST: (-1, 0),
            }
            dx, dy = DIRECTION_VECTORS[direction]

            nextState = (int(x + dx), int(y + dy))

            if not self.walls[nextState[0]][nextState[1]]:  # Only allow valid moves
                successors.append((nextState, direction, 1))  # Cost = 1 per step

        return successors


class FoodSearchProblem:
    def __init__(self, gameState, start, foodList):
        self.start = start
        self.foodList = foodList
        self.walls = gameState.get_walls()

    def getStartState(self):
        return self.start

    def isGoalState(self, state):
        return state in self.foodList  # Goal is any food position

    def getSuccessors(self, state):
        successors = []
        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            DIRECTION_VECTORS = {
                Directions.NORTH: (0, 1),
                Directions.SOUTH: (0, -1),
                Directions.EAST: (1, 0),
                Directions.WEST: (-1, 0),
            }
            dx, dy = DIRECTION_VECTORS[direction]

            nextState = (int(x + dx), int(y + dy))

            if not self.walls[nextState[0]][nextState[1]]:  # Only allow valid moves
                successors.append((nextState, direction, 1))  # Cost = 1 per step

        return successors





class OffensiveReflexAgent(ReflexCaptureAgent):
    """
      A reflex agent that seeks food. This is an agent
      we give you to get an idea of what an offensive agent might look like,
      but it is by no means the best or only way to build an offensive agent.
      """

    def __init__(self, index):
        super().__init__(index)
        self.previous_food_positions = []
        self.target_food = None
        self.test_1_scared = True
        self.test_0_scared = True

    def breadthFirstSearch(self, problem):
        """Search for the shortest path to a goal state using BFS."""
        queue = Queue()
        visited = set()

        start_state = problem.getStartState()
        queue.push((start_state, []))  # (current position, path taken)

        while not queue.is_empty():
            current_state, path = queue.pop()

            if problem.isGoalState(current_state):
                return path  # Return the shortest path to the goal

            if current_state not in visited:
                visited.add(current_state)

                for next_state, action, cost in problem.getSuccessors(current_state):
                    if next_state not in visited:
                        queue.push((next_state, path + [action]))  # Extend path

        return []  # Return empty path if no solution is found

    def choose_action(self, game_state):
        """
        Picks the best action based on multiple scoring factors.
        Switches to defensive behavior when the team is winning.
        """

        # my_state = game_state.get_agent_state(self.index)
        #
        # if not my_state.is_pacman:
        #     self.max_food = self.carried_food + self.food_left


        # Check the current score
        current_score = self.get_score(game_state)
        print(f"3: this is the current score:{current_score}")

        self.food_left = len(self.get_food(game_state).as_list())
        # if self.max_food == 30:
        #     self.carried_food = max(0, 30 - self.food_left)
        # else: self.carried_food = max(0, (self.max_food - self.food_left))
        #self.carried_food = (self.max_food or self.food_left) - self.food_left
        self.carried_food = max(0, (self.max_food - self.food_left))

        # If we're winning (score > 0), use defensive behavior
        if current_score > 0:
            print("3: Im am defending now because we are winning")
            return self.choose_defensive_action(game_state)

        else:
            print("3: Im am attacking now because we are losing")
            return self.choose_offensive_action(game_state)



    def choose_defensive_action(self, game_state):
        """
        Implements defensive strategy when the team is winning.
        This uses logic similar to the DefensiveReflexAgent.
        """

        # print(f"I'm offensive on red: {game_state.is_on_red_team(self.index)}
        game_state.is_on_red_team(self.index)
        print(f"this is my index of the offensive defensif {self.index}")
        #print("now I am a defensif agent")
          # Store previous food locations
        print(
            f"3 D: Carried food: {self.carried_food}, Food left: {self.food_left},max food: {self.max_food}, Score diff: {self.get_score(game_state)}")

        # self.carried_food = 0
        self.max_food = self.food_left

        legalActions = game_state.get_legal_actions(self.index)

        if not legalActions:
            return Directions.STOP  # Stop if no moves available

        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)  # Remove STOP if other moves exist

        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        # suc_food_list = self.get_food(self.get_successor(game_state, Directions.STOP)).as_list()
        food_left = len(food_list)
        # suc_food_left = len(suc_food_list)
        enemies = self.get_opponents(game_state)
        successor = self.get_successor(game_state, legalActions[0])
        my_state = successor.get_agent_state(self.index)
        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height
        mid_pos = (layout_width // 2, layout_height // 2)
        our_capsule_positions = self.get_capsules_you_are_defending(game_state)

        len_list_capsule = len(our_capsule_positions)

        best_action = None
        best_score = float('-inf')

        # Get middle patrol line positions

        # Determine whether we're red or blue team to find the proper patrol line
        team_index = 1 if game_state.is_on_red_team(self.index) else -1
        patrol_x = (layout_width // 2) - team_index  # Adjust the patrol line based on team
        patrol_2x = (layout_width // 2) - (team_index*2)

        # Create list of patrol positions along the border
        patrol_positions = []
        for y in range(layout_height):

            if not game_state.has_wall(patrol_x, y):
                patrol_positions.append((patrol_x, y))

        if self.def_scared_time > 0:
            self.def_scared_time -= 1

        for action in legalActions:
            score = 0  # Start score for this action
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_position(self.index)

            # Get information about invaders
            enemies = self.get_opponents(game_state)
            # enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            # invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

            enemy_positions = [game_state.get_agent_position(enemy) for enemy in enemies if
                               game_state.get_agent_position(enemy) is not None]
            enemy_distances = [self.get_maze_distance(next_pos, enemy) for enemy in enemy_positions]
            # Strongly prioritize attacking invaders
            enemy_state = [game_state.get_agent_state(enemy) for enemy in enemies]

            # Strongly prioritize attacking invaders
            if self.def_scared_time == 0:
                for enemy in enemy_state:
                    if enemy_distances and enemy.is_pacman:
                        print("Im am the offensive that defend")
                        # invader_dists = [self.get_maze_distance(next_pos, a.get_position()) for a in invaders]
                        min_enemy_distance = min(enemy_distances)
                        score += 10000 / (min_enemy_distance + 1)  # Higher priority as invaders get closer
                        if min_enemy_distance == 0:
                            score += 100000

            current_food = self.get_food_you_are_defending(game_state)
            current_food_positions = [(x, y) for x in range(current_food.width)
                                      for y in range(current_food.height) if current_food[x][y]]

            # Detect missing food (i.e., food that was in the previous state but not anymore)
            missing_food_positions = list(set(self.previous_food_positions) - set(current_food_positions))

            # Update previous food positions
            self.previous_food_positions = current_food_positions

            my_pos = game_state.get_agent_position(self.index)


            if missing_food_positions:
                # Lock onto the first missing food if we don't have a target
                if self.target_food is None or self.target_food not in current_food_positions:
                    self.target_food = missing_food_positions[0]  # Set a food target

            print(f"Current food target: {self.target_food}")
            path_to_food = False
            # Continue following the path to the target food
            if self.target_food:
                path_to_food = self.breadthFirstSearch(
                    CapsuleSearchProblem(game_state, next_pos, self.target_food)
                )

            # if path_to_food:
            #     print(f"Following path to missing food: {path_to_food}")
            #     chosen_action = path_to_food[0] # First move in BFS path
            #     lenght_path = len(path_to_food)
            #
            #     # Check if chosen action is valid
            #     legalActions = game_state.get_legal_actions(self.index)
            #     if chosen_action not in legalActions:
            #         print(f"WARNING: Illegal action {chosen_action}, selecting alternative.")
            #         return random.choice(legalActions)  # Pick a safe fallback
            #
            #
            #     return chosen_action  # Follow BFS path
            #     #score += 10000 / (lenght_path + 1)
            #
            # else:
            #     print("No path found, clearing target.")
            #     self.target_food = None  # Reset the target if we can't reach it

            if path_to_food and self.def_scared_time == 0:
                print(f"Following path to missing food: {path_to_food}")
                chosen_action = path_to_food[0]  # First move in BFS path
                lenght_path = len(path_to_food)

                # Check if chosen action is valid
                legalActions = game_state.get_legal_actions(self.index)
                if chosen_action not in legalActions:
                    print(f"WARNING: Illegal action {chosen_action}, selecting alternative.")
                    return random.choice(legalActions)  # Pick a safe fallback

                if enemy_distances and lenght_path <= 6:
                    print("im am chasing the enemy")
                    self.target_food = None
                    # invader_dists = [self.get_maze_distance(next_pos, a.get_position()) for a in invaders]
                    min_enemy_distance = min(enemy_distances)
                    print(f"this is the min enemy distance {min_enemy_distance}")
                    score += 100000 / (min_enemy_distance + 1)  # Higher priority as invaders get closer
                    if min_enemy_distance == 1:
                        next_pos = enemy_positions
                        print("I am near to the enemy")
                    if my_pos in enemy_positions:
                        print("i have killed the enemy")
                        score += 1000000
                else:
                    return chosen_action
                # return chosen_action  # Follow BFS path
                # score += 100000 / (lenght_path + 1)

            #there's no food or we're scared
            else:
                for enemy in enemy_state:
                    #enemy is not offensive and hasn't eaten anything in our camp
                    if not missing_food_positions and not enemy.is_pacman and not path_to_food:
                        print("OD: Patrolling midlane...")
                        self.patrolling.add(game_state.get_agent_state(self.index))

                        teamMatePos = None
                        if game_state.is_on_red_team(self.index):
                            teamMatePos = game_state.get_agent_position(2)
                            print(teamMatePos)
                        else:
                            teamMatePos = game_state.get_agent_position(3)
                            print(teamMatePos)

                        patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]

                        if patrol_dists:
                            print("im going to the patrol positions")
                            min_patrol_dist = min(patrol_dists)
                            score += 1000 / (min_patrol_dist + 1)  # Weak patrol incentive

                        if teamMatePos in patrol_positions:
                            print("both patrolling")
                            # strategically keep distance from each other
                            distanceToTeamMate = self.get_maze_distance(next_pos, teamMatePos)
                            if distanceToTeamMate <= 7:
                                score -= 2000 / (distanceToTeamMate + 1)

                    #enemy is offensive, eating our food
                    else:
                        if game_state.get_agent_state(self.index) in self.patrolling:
                            self.patrolling.remove(game_state.get_agent_state(self.index))

                        print("Skipping patrol, prioritizing missing food or enemies.")  # Weak patrol incentive

            # for enemy in enemy_state:
            #     if not missing_food_positions and not enemy.is_pacman and not path_to_food:
            #         print("OD: Patrolling midlane...")
            #         self.patrolling.add(game_state.get_agent_state(self.index))
            #
            #
            #         teamMatePos = None
            #         if game_state.is_on_red_team(self.index):
            #             teamMatePos = game_state.get_agent_position(2)
            #             print(teamMatePos)
            #         else:
            #             teamMatePos = game_state.get_agent_position(3)
            #             print(teamMatePos)
            #
            #         patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]
            #
            #         if patrol_dists:
            #             print("im going to the patrol positions")
            #             min_patrol_dist = min(patrol_dists)
            #             score += 1000 / (min_patrol_dist + 1)  # Weak patrol incentive
            #
            #
            #         if teamMatePos in patrol_positions:
            #             print("both patrolling")
            #             #strategically keep distance from each other
            #             distanceToTeamMate = self.get_maze_distance(next_pos,teamMatePos)
            #             if distanceToTeamMate <= 7:
            #                 score -= 2000/(distanceToTeamMate + 1)
            #
            #     else:
            #         if game_state.get_agent_state(self.index) in self.patrolling:
            #             self.patrolling.remove(game_state.get_agent_state(self.index))
            #
            #         print("Skipping patrol, prioritizing missing food or enemies.") # Weak patrol incentive

            if len_list_capsule == 1 and self.test_1_scared:
                self.def_scared_time = 40
                self.test_1_scared = False

            if len_list_capsule == 0 and self.test_0_scared:
                self.def_scared_time = 40
                self.test_0_scared = False

            #died, reinitialize scared timer
            if next_pos == self.start:
                score -= 1000000
                self.def_scared_time = 0

            #we're scared of our enemy
            if self.def_scared_time > 0:
                for enemi in enemy_state:
                    if enemy_distances and enemi.is_pacman:
                        print("im am scaared of the enemy")
                        # invader_dists = [self.get_maze_distance(next_pos, a.get_position()) for a in invaders]
                        min_enemy_distance = min(enemy_distances)
                        score -= 1000 / (min_enemy_distance + 1)  # Higher priority as invaders get closer
                        print(f"this is the min enemy sitance {min_enemy_distance}")
                        # if next_pos == self.start:
                        #     print("ik ben dood")
                        #     score -= 1000000
                        #     self.def_scared_time = 0

                        #safe distance
                        if  3 <= min_enemy_distance <= 5:
                            print("I stay at a good distance of the enemy")
                            score += 100000

            #we're scared but can't see the enemy, so patrol
            if self.def_scared_time > 0 and not enemy_distances:
                patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]
                self.target_food = None
                if patrol_dists:
                    min_patrol_dist = min(patrol_dists)
                    score += 1000 / (min_patrol_dist + 1)  # Weak patrol incentive
            # teamMatePos = None
            # if game_state.is_on_red_team(self.index):
            #     teamMatePos = game_state.get_agent_position(2)
            #     print(teamMatePos)
            # else:
            #     teamMatePos = game_state.get_agent_position(3)
            #     print(teamMatePos)
            # if len_list_capsule == 1 and teamMatePos in patrol_positions:
            #     our_capsule_path = self.breadthFirstSearch(BFSSearchProblem(game_state, next_pos, our_capsule_positions))
            #     chosen_action = our_capsule_path[0]
            #     legalActions = game_state.get_legal_actions(self.index)
            #     if chosen_action not in legalActions:
            #         print(f"WARNING: Illegal action {chosen_action}, selecting alternative.")
            #         return random.choice(legalActions)  # Pick a safe fallback
            #     return chosen_action
            # if len_list_capsule == 1 and next_pos in our_capsule_positions:
            #     best_action = Directions.STOP
            #     return best_action
            #


            # Penalize staying still

            # if action == Directions.STOP:
            #     score -= 100

            # # Penalize reversing
            # rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
            # if action == rev:
            #     score -= 50

            # Choose the best action
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            return random.choice(legalActions)

        return best_action

    def choose_offensive_action(self, game_state):
        """
        Original offensive behavior for when the team is losing or tied.
        """
        print("now I am an offensive agent")

        legalActions = game_state.get_legal_actions(self.index)

        if not legalActions:
            return Directions.STOP  # Stop if no moves available

        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)  # Remove STOP if other moves exist

        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        suc_food_list = self.get_food(self.get_successor(game_state, Directions.STOP)).as_list()
        food_left = len(food_list)
        suc_food_left = len(suc_food_list)
        enemies = self.get_opponents(game_state)
        successor = self.get_successor(game_state, legalActions[0])
        my_state = successor.get_agent_state(self.index)
        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height
        mid_pos = (layout_width // 2, layout_height // 2)
        # print(f"this is the timer {self.scared_time}")

        team_index = 1 if game_state.is_on_red_team(self.index) else -1
        patrol_x = (layout_width // 2) - team_index
        safe_border_positions = [(patrol_x, pos) for pos in range(layout_height)]
        capsule_positions = self.get_capsules(game_state)
        our_capsule_positions = self.get_capsules_you_are_defending(game_state)

        best_action = None
        best_score = float('-inf')

        if self.scared_time > 0:
            self.scared_time -= 1
        print(f"this is the offensive agent timer:{self.scared_time}")
            # print("timer is decreasing with 1")
            # print(f"{self.scared_time}")

        for action in legalActions:
            score = 0  # Start score for this action

            successor = self.get_successor(game_state, action)
            my_state = successor.get_agent_state(self.index)
            next_pos = successor.get_agent_position(self.index)
            # lenght_path_mid = len(self.breadthFirstSearch(CapsuleSearchProblem(game_state, next_pos, mid_pos)))
            # print(lenght_path_mid)
            # 1. Distance to food matters
            food_distances = [self.get_maze_distance(next_pos, food) for food in food_list]
            if food_distances:
                min_food_dist = min(food_distances)
                score += 100 / (min_food_dist + 1)  # Encourage moving towards food

            # 2. Avoiding enemies matters
            enemy_positions = [game_state.get_agent_position(enemy) for enemy in enemies if
                               game_state.get_agent_position(enemy) is not None]

            enemy_distances = [self.get_maze_distance(next_pos, enemy) for enemy in enemy_positions]

            #enemy is close, I'm a pacman and enemy is defensive
            if enemy_distances and self.scared_time <= 12 and my_state.is_pacman:
                min_enemy_dist = min(enemy_distances)
                print("I am here, i'm going to attack ")
                if min_enemy_dist <= 2:
                    score -= 100000  # VERY bad if enemy is too close
                elif min_enemy_dist <= 5:
                    score -= 2000 / (min_enemy_dist + 1)
                else:
                    score += 100  # Safe exploration is reward


            #enemy is close and we can kill, go after them
            if self.scared_time > 12 and my_state.is_pacman and enemy_distances:
                min_enemy_dist = min(enemy_distances)
                print("I am here, i'm going to kill the enemy ")
                if min_enemy_dist <= 3:
                    score += 100000/(min_enemy_dist+ 1)  # VERY good if enemy is too close

            # if not my_state.is_pacman:
            #     print("I set carried food to 0")
            #     self.carried_food = 0

            # 3. Returning to base when necessary
            # self.max_food = 30
            # carried_food = max_food - food_left  # Estimate how much food is carried

            # if self.get_score(game_state) < 0:
            #     carried_food = 0

            print(
                f"3: Carried food: {self.carried_food}, Food left: {self.food_left},max food: {self.max_food}, Score diff: {self.get_score(game_state)}")

            if self.carried_food >= 6 and self.scared_time == 0:
                min_distance = float('inf')  # Start with a very large number
                home_pos = None  # Store the closest position

                for pos in safe_border_positions:
                    try:
                        dist = self.get_maze_distance(next_pos, pos)
                        if dist < min_distance:
                            min_distance = dist
                            home_pos = pos
                    except Exception:
                        print(f"Skipping invalid position: {pos}")
                home_dist = self.get_maze_distance(next_pos, home_pos)
                score += 2000/ (home_dist +1)
                if next_pos in home_pos:
                    score += 100000
                    # print("I have set the carried food to 0")
                    # self.carried_food = 0
                    # self.max_food = self.food_left

            if my_pos in safe_border_positions:
                print("I have set the carried food to 0")
                self.carried_food = 0
                self.max_food = self.food_left


            capsule_distances = [self.get_maze_distance(next_pos, capsule) for capsule in capsule_positions]

            # if capsule_positions:
            #     print(f"Capsules available: {capsule_positions}")
            # else:
            #     print("No capsules left.")
            if capsule_positions:

                if next_pos in capsule_positions:
                    score += 100000
                    self.scared_time = 40
                    # print(f"updated scared timer: {self.scared_time}")



            # if len(capsule_positions) == 1 and self.scared_time > 0:
            #     capsule_path = self.breadthFirstSearch(CapsuleSearchProblem(game_state, next_pos, capsule_positions[0]))
            #     pathLength = len(capsule_path)
            #     legalActions = game_state.get_legal_actions(self.index)
            #
            #     if pathLength == self.scared_time:
            #         print("pathlength is the same of the scared timer ")
            #         action = capsule_path[0]
            #         if action not in legalActions:
            #             print(f"WARNING: Illegal action {action}, selecting alternative.")
            #             return random.choice(legalActions)  # Pick a safe fallback
            #         else: return action




            if capsule_positions:
                #dichtsbijzijnde capsule
                closest_capsule = min(capsule_positions, key=lambda c: self.get_maze_distance(next_pos, c))
                capsule_path = self.breadthFirstSearch(CapsuleSearchProblem(game_state, next_pos, closest_capsule))

                if capsule_path:
                    #:) kortste pad tot capsule
                    score += 1000 / (len(capsule_path) + 1)  # Increase priority of capsules

                    # If an enemy is close and already has 2 food, make capsules even more valuable
                    if enemy_distances:
                        min_enemy_dist = min(enemy_distances)
                        if min_enemy_dist < 6 and self.carried_food >= 2:
                            score += 3000 / (len(capsule_path) + 1)
                    else: score += 100

            #Tijd over en nog niet veel voedsel verzamelt, blijf eten
            if self.scared_time > 12 and self.carried_food <= 12:
                print(self.scared_time)
                best_food_path = self.breadthFirstSearch(FoodSearchProblem(game_state, next_pos, food_list))
                print(f"food path found: {best_food_path}")
                if best_food_path == next_pos:
                    score += 10000# Strongly encourage taking the best food path
            #Genoeg voedsel gegeten en niet veilig
            if self.carried_food > 12 and self.scared_time > 0:
                min_distance = float('inf')  # Start with a very large number
                home_pos = None  # Store the closest position

                for pos in safe_border_positions:
                    try:
                        dist = self.get_maze_distance(next_pos, pos)
                        if dist < min_distance:
                            min_distance = dist
                            home_pos = pos
                    except Exception:
                        print(f"Skipping invalid position: {pos}")
                home_dist = self.get_maze_distance(next_pos, home_pos)
                score += 10000 / (home_dist + 1)

            #Niet meer veel tijd over voordat enemy niet meer scared is + genoeg verzameld, come back
            if 1 <= self.scared_time <= 12 and self.carried_food >= 6:
                print("er zijn niet meer veel moves mogelijk")
                home_dist = self.get_maze_distance(self.start, next_pos)
                score += 10000 / (home_dist + 1)

            #Enemy niet meer scared, geen powerpellets, keer terug naar bordr
            if self.scared_time == 0 and not capsule_positions and self.carried_food >1:
                # home_dist = self.get_maze_distance(self.start, next_pos)
                # score += 10000 / (home_dist + 1)
                min_distance = float('inf')  # Start with a very large number
                home_pos = None  # Store the closest position

                for pos in safe_border_positions:
                    try:
                        dist = self.get_maze_distance(next_pos, pos)
                        if dist < min_distance:
                            min_distance = dist
                            home_pos = pos
                    except Exception:
                        print(f"Skipping invalid position: {pos}")
                home_dist = self.get_maze_distance(next_pos, home_pos)
                score += 10000 / (home_dist + 1)

            if self.food_left <= 2:
                home_dist = self.get_maze_distance(self.start, next_pos)
                score += 3000 / (home_dist + 1)

            # 4. Choosing the best action
            if score > best_score:
                best_score = score
                best_action = action

        # if there are no actions (what is impossible normaly) choose a random action
        if best_action is None:
            return random.choice(legalActions)

        return best_action



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def __init__(self, index):
        super().__init__(index)
        self.previous_food_positions = []
        self.target_food = None
        self.test_1_scared = True
        self.test_0_scared = True

    def breadthFirstSearch(self, problem):
        """Search for the shortest path to a goal state using BFS."""
        queue = Queue()
        visited = set()

        start_state = problem.getStartState()
        queue.push((start_state, []))  # (current position, path taken)

        while not queue.is_empty():
            current_state, path = queue.pop()

            if problem.isGoalState(current_state):
                return path  # Return the shortest path to the goal

            if current_state not in visited:
                visited.add(current_state)

                for next_state, action, cost in problem.getSuccessors(current_state):
                    if next_state not in visited:
                        queue.push((next_state, path + [action]))  # Extend path

        return []  # Return empty path if no solution is found

    def choose_action(self, game_state):
        """
        Implements defensive strategy when the team is winning.
        This uses logic similar to the DefensiveReflexAgent.
        """

          # Store previous food locations

        legalActions = game_state.get_legal_actions(self.index)
        print(f"start{self.start}")
        if not legalActions:
            return Directions.STOP  # Stop if no moves available

        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)  # Remove STOP if other moves exist

        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        suc_food_list = self.get_food(self.get_successor(game_state, Directions.STOP)).as_list()
        food_left = len(food_list)
        suc_food_left = len(suc_food_list)
        enemies = self.get_opponents(game_state)
        successor = self.get_successor(game_state, legalActions[0])
        my_state = successor.get_agent_state(self.index)
        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height
        mid_pos = (layout_width // 2, layout_height // 2)
        our_capsule_positions = self.get_capsules_you_are_defending(game_state)

        best_action = None
        best_score = float('-inf')

        # Get middle patrol line positions

        # Determine whether we're red or blue team to find the proper patrol line
        team_index = 1 if game_state.is_on_red_team(self.index) else -1
        patrol_x = (layout_width // 2) - team_index  # Adjust the patrol line based on team

        # Create list of patrol positions along the border
        patrol_positions = []
        for y in range(layout_height):

            if not game_state.has_wall(patrol_x, y):
                patrol_positions.append((patrol_x, y))

        if self.our_scared_time > 0:
            self.our_scared_time -= 1

        print(f"this is our scared timer: {self.our_scared_time}")

        my_pos = game_state.get_agent_position(self.index)

        for action in legalActions:
            score = 0  # Start score for this action
            successor = self.get_successor(game_state, action)
            next_pos = successor.get_agent_position(self.index)

            # Get information about invaders
            enemies = self.get_opponents(game_state)
            # enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
            # invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]

            enemy_positions = [game_state.get_agent_position(enemy) for enemy in enemies if
                               game_state.get_agent_position(enemy) is not None]
            enemy_distances = [self.get_maze_distance(next_pos, enemy) for enemy in enemy_positions]
            enemy_state = [game_state.get_agent_state(enemy) for enemy in enemies]
            # print(f"this are the enemies:{enemies}")
            # print(f"this are the enemie state:{enemy_state}")
            # Strongly prioritize attacking invaders


            if self.our_scared_time == 0:
                for enemy in enemy_state:
                    #Zien de enemy en hij is vulnerable, val aan
                    if enemy_distances and enemy.is_pacman:
                        print("im am chasing the enely")
                        # invader_dists = [self.get_maze_distance(next_pos, a.get_position()) for a in invaders]
                        min_enemy_distance = min(enemy_distances)
                        print(f"this is the minnnn enemy dis {min_enemy_distance}")
                        score += 10000 / (min_enemy_distance + 1)  # Higher priority as invaders get closer
                        if min_enemy_distance == 0:
                            print("i have killed the enemy")
                            score += 1000000

            current_food = self.get_food_you_are_defending(game_state)
            current_food_positions = [(x, y) for x in range(current_food.width)
                                      for y in range(current_food.height) if current_food[x][y]]

            # Detect missing food (i.e., food that was in the previous state but not anymore)
            missing_food_positions = list(set(self.previous_food_positions) - set(current_food_positions))

            # Update previous food positions
            self.previous_food_positions = current_food_positions

            my_pos = game_state.get_agent_position(self.index)

            len_list_capsule = len(our_capsule_positions)
            print(f"length capsules: {len_list_capsule}")
            print(f"this are our capsules: {our_capsule_positions}")


            

            if missing_food_positions:
                self.target_food = missing_food_positions[0]
                # Lock onto the first missing food if we don't have a target
                #if self.target_food is None or self.target_food not in current_food_positions:
                    #self.target_food = missing_food_positions[0]  # Set a food target

                print(f"Current food target: {self.target_food}")
                print(f"missing food is {missing_food_positions}")


            path_to_food = False
            # Continue following the path to the target food
            if self.target_food:
                path_to_food = self.breadthFirstSearch(
                    CapsuleSearchProblem(game_state, my_pos, self.target_food)
                )


            if path_to_food and self.our_scared_time == 0:
                print(f"Following path to missing food: {path_to_food}")
                chosen_action = path_to_food[0]  # First move in BFS path
                lenght_path = len(path_to_food)

                # Check if chosen action is valid
                legalActions = game_state.get_legal_actions(self.index)
                if chosen_action not in legalActions:
                    print(f"WARNING: Illegal action {chosen_action}, selecting alternative.")
                    return random.choice(legalActions)  # Pick a safe fallback

                if enemy_distances and lenght_path <= 5:
                    print("im am chasing the enemy")
                    # invader_dists = [self.get_maze_distance(next_pos, a.get_position()) for a in invaders]
                    min_enemy_distance = min(enemy_distances)
                    print(f"this is the min enemy distance {min_enemy_distance}")
                    score += 10000 / (min_enemy_distance + 1)  # Higher priority as invaders get closer
                    if min_enemy_distance <= 3: #added this to test
                        self.target_food = None
                        score += 100000 / (min_enemy_distance + 1)
                    if min_enemy_distance == 1:
                        next_pos = enemy_positions
                        print("I am near to the enemy")
                    if my_pos == enemy_positions:
                        print("i have killed the enemy")
                        score += 1000000
                else:
                    return chosen_action





                # return chosen_action  # Follow BFS path
                # score += 100000 / (lenght_path + 1)
            else:
                for enemy in enemy_state:
                    if not missing_food_positions and not enemy.is_pacman and not path_to_food:
                        print("Patrolling midlane...")
                        self.patrolling.add(game_state.get_agent_state(self.index))

                        teamMatePos = None
                        if game_state.is_on_red_team(self.index):
                            teamMatePos = game_state.get_agent_position(0)
                            print(teamMatePos)
                        else:
                            teamMatePos = game_state.get_agent_position(1)
                            print(teamMatePos)

                        patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]

                        if patrol_dists:
                            print("im going to the patrol positions")
                            min_patrol_dist = min(patrol_dists)
                            score += 1000 / (min_patrol_dist + 1)  # Weak patrol incentive

                        if teamMatePos in patrol_positions:
                            print("both patrolling")
                            # strategically keep distance from each other
                            distanceToTeamMate = self.get_maze_distance(next_pos, teamMatePos)
                            if distanceToTeamMate <= 7:
                                score -= 2000 / (distanceToTeamMate + 1)

                    else:
                        if game_state.get_agent_state(self.index) in self.patrolling:
                            self.patrolling.remove(game_state.get_agent_state(self.index))

                        print("Skipping patrol, prioritizing missing food or enemies.")  # Weak patrol incentive



                # for enemy in enemy_state:
                #     if not missing_food_positions and not enemy.is_pacman and not path_to_food:
                #         self.patrolling.add(game_state.get_agent_state(self.index))
                #
                #         print("Patrolling midlane...")
                #         patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]
                #         self.target_food = None
                #         if patrol_dists:
                #             min_patrol_dist = min(patrol_dists)
                #             score += 1000 / (min_patrol_dist + 1)  # Weak patrol incentive
                #     else:
                #         if game_state.get_agent_state(self.index) in self.patrolling:
                #            self.patrolling.remove(game_state.get_agent_state(self.index))
                #         print("Skipping patrol, prioritizing missing food or enemies.")  # Weak patrol incentive
                #



            if len_list_capsule == 1 and self.test_1_scared:
                print("ik zit in die loop")
                self.our_scared_time = 40
                print(f"here is Our scared timer:{self.our_scared_time} ")
                self.test_1_scared = False
            if len_list_capsule == 0 and self.test_0_scared:
                self.our_scared_time = 40
                self.test_0_scared = False

            print(f" this is my next pos {next_pos}")
            if next_pos == self.start:
                print("yeeeessss")

            # for enemy in enemy_state:
            if self.our_scared_time > 0:
                for enemi in enemy_state:
                    if enemy_distances and enemi.is_pacman:
                        print("im am scaared of the enemy")
                        # invader_dists = [self.get_maze_distance(next_pos, a.get_position()) for a in invaders]
                        min_enemy_distance = min(enemy_distances)
                        score -= 1000 / (min_enemy_distance + 1)  # Higher priority as invaders get closer
                        print(f"this is the min enemy sitance {min_enemy_distance}")
                        if next_pos == self.start:
                            print("ik ben dood")
                            score -= 1000000
                            self.our_scared_time = 0

                        if  3 <= min_enemy_distance <= 5:
                            print("I stay at a good distance of the enemy")
                            score += 100000

            if self.our_scared_time > 0 and not enemy_distances:
                print("I am scared but enemy is too far, Patrolling midlane...")
                patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]
                self.target_food = None
                if patrol_dists:
                    min_patrol_dist = min(patrol_dists)
                    score += 1000 / (min_patrol_dist + 1)  # Weak patrol incentive


            # # Penalize staying still
            # if action == Directions.STOP:
            #     score -= 100


            # Choose the best action
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None:
            return random.choice(legalActions)

        return best_action
