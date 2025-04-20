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

#final code

# import util
import random
from capture_agents import CaptureAgent
from game import Directions
from util import nearest_point
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

        self.patrolling = set() #houdt bij welke agents patrouilleren op de midlane

    def register_initial_state(self, game_state): #beginstate en positie
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)


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

    def breadthFirstSearch(self, problem):
        """Search for the shortest path to a goal state using BFS. Given a search problem"""
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

    def returnHome(self,next_pos,factor,safe_border_positions,tempscore):
        """find the path to return home"""
       #keer terug  naar huis

        min_distance = float('inf')  # Start with a very large number
        home_pos = None  # Store the closest position
        for pos in safe_border_positions:
            try:
                dist = self.get_maze_distance(next_pos, pos)
                if dist < min_distance:
                    min_distance = dist
                    home_pos = pos
            except Exception:
                print(f"Skipping invalid position of the midlane: {pos}")
        home_dist = self.get_maze_distance(next_pos, home_pos)
        tempscore += factor / (home_dist + 1)

        if next_pos == home_pos:
            tempscore += 100000

        return tempscore



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



class OffensiveReflexAgent(ReflexCaptureAgent):
    """
        offensive agent that is split into to different agents, one offensive when the score is zero or negative
        and one defensive agent when we are winning.
      """

    def __init__(self, index):
        super().__init__(index)
        self.carried_food = 0
        self.food_left = 0 #zal bijhouden hoeveel capsules er bij de enemy overblijven
        self.max_food = 30  # elk team heeft 30 voedselcapsules in het begin
        self.previous_food_positions = [] #zal capsuleposities bijhouden van de vorige game state
        self.target_food = None  #positie van een voedselcapsule waar de agent naartoe moet lopen



    def choose_action(self, game_state):
        """
        Picks the best action based on multiple scoring factors.
        Switches to defensive behavior when the team is winning.
        """

        # Check the current score
        current_score = self.get_score(game_state)

        self.food_left = len(self.get_food(game_state).as_list()) #hoeveel capsules in huidige game_state overblijven bij de enemy
        self.carried_food = max(0, (self.max_food - self.food_left)) #hoeveel capsules er door ons werden opgegeten

        # If we're winning (score > 0), use defensive behavior
        if current_score > 0:
            return self.choose_defensive_action(game_state)

        #We're losing, behave offensive
        else:
            return self.choose_offensive_action(game_state)

    def choose_defensive_action(self, game_state):
        """
        Implements defensive strategy when the team is winning.
        This uses logic similar to the DefensiveReflexAgent.
        """


        self.max_food = self.food_left #update
        legalActions = game_state.get_legal_actions(self.index)
        scared_timer = game_state.get_agent_state(self.index).scared_timer
        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height
        team_index = 1 if game_state.is_on_red_team(self.index) else -1 # Determine whether we're red or blue team to find the proper patrol line

        if not legalActions:
            return Directions.STOP  # Stop if no moves available

        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)  # Remove STOP if other moves exist

        patrol_x = (layout_width // 2) - team_index  # Adjust the patrol line based on team
        # Create list of patrol positions along the border (without the walls)
        patrol_positions = []
        for y in range(layout_height):

            if not game_state.has_wall(patrol_x, y):
                patrol_positions.append((patrol_x, y))

        #we gaan de beste actie bepalen obv de score die wij geven aan een opvolgerstate
        best_action = None
        best_score = float('-inf')



        for action in legalActions:
            score = 0  # Start score for this action
            successor = self.get_successor(game_state, action) #succesorstate
            next_pos = successor.get_agent_position(self.index)

            # Get information about invaders
            enemies = self.get_opponents(game_state) #lijst v indices v opponents
            enemy_positions = [game_state.get_agent_position(enemyindex) for enemyindex in enemies if
                               game_state.get_agent_position(enemyindex) is not None] #list of positions of opponents
            enemy_distances = [self.get_maze_distance(next_pos, enemy) for enemy in enemy_positions] #list of distances to every opponent
            enemy_states = [game_state.get_agent_state(enemyidx) for enemyidx in enemies] #list of enemy states

            current_food = self.get_food_you_are_defending(game_state)  # matrix van capsules die we defenden waarbij de elementen booleans zijn
            current_food_positions = [(x, y) for x in range(current_food.width)
                                      for y in range(current_food.height) if
                                      current_food[x][y]]  # posities van voedselcapsules die wij defenden

            # posities van de voedselcapsules die niet meer in deze game state zitten
            missing_food_positions = list(set(self.previous_food_positions) - set(current_food_positions))

            # Update previous food positions to food positions of current state
            self.previous_food_positions = current_food_positions

            my_pos = game_state.get_agent_position(self.index)
            # Strongly prioritize attacking invaders
            if scared_timer == 0:
                for enemy in enemy_states:
                    #there are enemies we can see and they're offensive
                    if enemy_distances and enemy.is_pacman:
                        min_enemy_distance = min(enemy_distances) #pick the closest enemy
                        score += 10000 / (min_enemy_distance + 1)  #Prioritise getting closer to enemies
                        if min_enemy_distance == 0:
                            score += 100000


            #er werd voedsel opgegeten door enemy
            if missing_food_positions:
                # Look onto the first missing food if we don't have a target
                if self.target_food is None or self.target_food not in current_food_positions:
                    self.target_food = missing_food_positions[0]  # Set a food target


            #construeer pad tot opgegeten voedsel
            path_to_food = False
            if self.target_food:
                path_to_food = self.breadthFirstSearch(
                    BFSSearchProblem(game_state, next_pos, self.target_food)
                )

            #pad gevonden tot opgegeten voedsel en opportuniteit om enemy te doden
            if path_to_food and scared_timer == 0:
                chosen_action = path_to_food[0]  # First move in BFS path
                lenght_path = len(path_to_food)

                # Check if chosen action is valid
                legalActions = game_state.get_legal_actions(self.index)
                if chosen_action not in legalActions:
                    return random.choice(legalActions)  # Pick a safe fallback

                #enemy iz zichtbaar
                if enemy_distances and lenght_path <= 5:
                    min_enemy_distance = min(enemy_distances)
                    score += 10000 / (min_enemy_distance + 1)  # Hoge beloning om dicht bij enemy te zijn
                    if min_enemy_distance <= 3:
                        self.target_food = None #Target food op none, zodat hij toch naar enemy zou gaan ipv pad afmaken
                        score += 100000 / (min_enemy_distance + 1)
                    if my_pos == enemy_positions:
                        score += 1000000 #enemy is dood
                else:
                    return chosen_action

            else: #niks opgegeten of scared_timer > 0
                for enemy in enemy_states:
                    if not missing_food_positions and not enemy.is_pacman and not path_to_food:
                        #zien de enemy niet, niks opgegeten
                        #laat de agent patrouilleren
                        self.patrolling.add(game_state.get_agent_state(self.index))

                        #positie van de teammate
                        teamMatePos = None
                        if game_state.is_on_red_team(self.index):
                            teamMatePos = game_state.get_agent_position(2)
                        else:
                            teamMatePos = game_state.get_agent_position(3)

                        patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]

                        if patrol_dists:
                            min_patrol_dist = min(patrol_dists)
                            score += 1000 / (min_patrol_dist + 1)

                        if teamMatePos in patrol_positions:
                            # strategically keep distance from each other
                            distanceToTeamMate = self.get_maze_distance(next_pos, teamMatePos)
                            if distanceToTeamMate <= 7:
                                score -= 2000 / (distanceToTeamMate + 1)

                    else:
                        if game_state.get_agent_state(self.index) in self.patrolling:
                            #stop met patrouilleren
                            self.patrolling.remove(game_state.get_agent_state(self.index))


            # die is bad
            if next_pos == self.start:
                score -= 1000000


            # we're scared of our enemy
            if scared_timer > 0:
                for enemi in enemy_states:
                    if enemy_distances and enemi.is_pacman:
                        min_enemy_distance = min(enemy_distances)
                        score -= 1000 / (min_enemy_distance + 1)  # Blijf weg van een aanvallende enemy

                        # Blijf op een korte ,veilige afstand van de enemy
                        if 3 <= min_enemy_distance <= 5:
                            score += 100000

            # we're scared but can't see the enemy, so patrol because we can do nothing more
            if scared_timer > 0 and not enemy_distances:
                patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]
                self.target_food = None
                if patrol_dists:
                    min_patrol_dist = min(patrol_dists)
                    score += 1000 / (min_patrol_dist + 1)  #Dichtsbijzijnde patol position

            # Choose the  action with the highest score
            if score > best_score:
                best_score = score
                best_action = action

        #Random actie als fallback
        if best_action is None:
            return random.choice(legalActions)

        return best_action

    def choose_offensive_action(self, game_state):
        """
        Offensive behavior for when the team is losing or tied.
        """
        scared_timer_enemy = []
        legalActions = game_state.get_legal_actions(self.index)

        if not legalActions:
            return Directions.STOP  # Stop if no moves available

        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)  # Remove STOP if other moves exist

        my_pos = game_state.get_agent_position(self.index)
        food_list = self.get_food(game_state).as_list()
        enemies = self.get_opponents(game_state)
        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height
        team_index = 1 if game_state.is_on_red_team(self.index) else -1
        patrol_x = (layout_width // 2) - team_index
        safe_border_positions = [(patrol_x, pos) for pos in range(layout_height)]
        capsule_positions = self.get_capsules(game_state) #capsules of opposite team

        for enemyidx in enemies:
            scared_timer_enemy.append(game_state.get_agent_state(enemyidx).scared_timer)


        best_action = None
        best_score = float('-inf')



        for action in legalActions:

            score = 0  # Start score for this action

            successor = self.get_successor(game_state, action)
            my_state = successor.get_agent_state(self.index)
            next_pos = successor.get_agent_position(self.index)
            # 1. Distance to food matters
            food_distances = [self.get_maze_distance(next_pos, food) for food in food_list]
            if food_distances:
                min_food_dist = min(food_distances)
                score += 100 / (min_food_dist + 1)  # Encourage moving towards food

            # 2. Avoiding enemies matters
            enemy_positions = [game_state.get_agent_position(enemy) for enemy in enemies if
                               game_state.get_agent_position(enemy) is not None]

            enemy_distances = [self.get_maze_distance(next_pos, enemy) for enemy in enemy_positions]


            for scared_timer_en in scared_timer_enemy:
                # enemy is close, I'm a pacman and enemy is defensive
                if enemy_distances and scared_timer_en <= 12 and my_state.is_pacman:
                    min_enemy_dist = min(enemy_distances)
                    if min_enemy_dist <= 2:
                        score -= 100000  #Enemy is niet voor lang scared, VERY bad if enemy is too close
                    elif min_enemy_dist <= 5:
                        score -= 2000 / (min_enemy_dist + 1)
                    else:
                        score += 100  # Safe exploration is reward

                # enemy is close and we have enough time to kill, go chase them
                if scared_timer_en > 12 and my_state.is_pacman and enemy_distances:
                    min_enemy_dist = min(enemy_distances)
                    if min_enemy_dist <= 3:
                        score += 100000 / (min_enemy_dist + 1)  # VERY good if enemy is too close

                #enemy became offensive and we've collected enough food according to us
                if self.carried_food >= 6 and scared_timer_en == 0:

                    # #Vind dichtsbijzijnde escape/toegang
                    score = self.returnHome(next_pos, 2000, safe_border_positions, score)

                #we zitten in onze kant van de grid
                if my_pos in safe_border_positions:
                    self.carried_food = 0
                    self.max_food = self.food_left #update het max aantal voedselcapsules dat je kan opeten aan de enemy's kant
                #grote beloning om enemy capsule op te eten
                if capsule_positions:
                    if next_pos in capsule_positions:
                        score += 100000


                #Er zijn nog powerpellets om op te eten?
                if capsule_positions:
                    # dichtsbijzijnde capsule van de enemy
                    closest_capsule = min(capsule_positions, key=lambda c: self.get_maze_distance(next_pos, c)) #map van maze_distance tussen succesor positie en positie van elke powerpellet, dan minimum nemen
                    #pad tot dichtsbijzijnde capsule
                    capsule_path = self.breadthFirstSearch(BFSSearchProblem(game_state, next_pos, closest_capsule))

                    if capsule_path:
                        #er is een kortste pad tot capsule
                        score += 1000 / (len(capsule_path) + 1)  # hoe korter het pad hoe groter de score

                        # If an enemy is close and I have collected at least 2 foods, ok make capsules more valuable
                        if enemy_distances:
                            min_enemy_dist = min(enemy_distances)
                            if min_enemy_dist < 6 and self.carried_food >= 2:
                                score += 3000 / (len(capsule_path) + 1)
                        else:
                            score += 100

                # Tijd over en nog niet veel voedsel verzamelt, blijf eten
                if scared_timer_en > 12 and self.carried_food <= 12:
                    for food in food_list:
                        best_food_path = self.breadthFirstSearch(BFSSearchProblem(game_state, next_pos, food))
                        if best_food_path:
                            if best_food_path[0] == next_pos:
                                score += 10000  # Strongly encourage taking the best food path

                # Genoeg voedsel gegeten en veilig
                if self.carried_food > 12 and scared_timer_en > 0:
                    #keer terug  naar huis
                    score = self.returnHome(next_pos, 10000, safe_border_positions, score)

                # Niet meer veel tijd over voordat enemy niet meer scared is + genoeg verzameld, come back
                if 1 <= scared_timer_en <= 12 and self.carried_food >= 6:
                    score = self.returnHome(next_pos, 10000, safe_border_positions, score)

                # Enemy niet meer scared, geen powerpellets, keer terug naar border
                if scared_timer_en == 0 and not capsule_positions and self.carried_food > 1:
                    score = self.returnHome(next_pos, 10000, safe_border_positions, score)
                # geen foods over bij enemy, jeeej we gaan winnen!
                if self.food_left <= 2:
                    score = self.returnHome(next_pos, 3000, safe_border_positions, score)

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
    A reflex defensive agent that tries to keep its side Pacman-free.
    """

    def __init__(self, index):
        super().__init__(index)
        self.previous_food_positions = []
        self.target_food = None




    def choose_action(self, game_state):


        legalActions = game_state.get_legal_actions(self.index)
        if not legalActions:
            return Directions.STOP  # Stop if no moves available

        if Directions.STOP in legalActions and len(legalActions) > 1:
            legalActions.remove(Directions.STOP)  # Remove STOP if other moves exist

        layout_width = game_state.data.layout.width
        layout_height = game_state.data.layout.height

        best_action = None
        best_score = float('-inf')

        #scared timer van de agent
        scared_timer = game_state.get_agent_state(self.index).scared_timer

        # Get middle patrol line positions
        # Determine whether we're red or blue team to find the proper patrol line
        team_index = 1 if game_state.is_on_red_team(self.index) else -1
        patrol_x = (layout_width // 2) - team_index  # Adjust the patrol line based on team

        # Create list of patrol positions along the border (without the walls)
        patrol_positions = []
        for y in range(layout_height):

            if not game_state.has_wall(patrol_x, y):
                patrol_positions.append((patrol_x, y))


        for action in legalActions:
            score = 0  # Start score for this action
            successor = self.get_successor(game_state, action) #succesorstate
            next_pos = successor.get_agent_position(self.index)

            enemies = self.get_opponents(game_state)

            # Get information about invaders
            enemy_positions = [game_state.get_agent_position(enemy) for enemy in enemies if
                               game_state.get_agent_position(enemy) is not None]
            enemy_distances = [self.get_maze_distance(next_pos, enemy) for enemy in enemy_positions]
            enemy_state = [game_state.get_agent_state(enemy) for enemy in enemies]

            # Strongly prioritize attacking invaders
            if scared_timer == 0:
                for enemy in enemy_state:
                    # Zien de enemy en hij is vulnerable, val aan
                    if enemy_distances and enemy.is_pacman:
                        min_enemy_distance = min(enemy_distances) #pick the closest enemy
                        score += 10000 / (min_enemy_distance + 1)  #Prioritise getting closer to enemies
                        if min_enemy_distance == 0:
                            score += 1000000

            current_food = self.get_food_you_are_defending(game_state) #matrix van capsules die we defenden waarbij de elementen booleans zijn
            current_food_positions = [(x, y) for x in range(current_food.width)
                                      for y in range(current_food.height) if current_food[x][y]] #posities van voedselcapsules die wij defenden
            
            #posities van de voedselcapsules die niet meer in deze game state zitten
            missing_food_positions = list(set(self.previous_food_positions) - set(current_food_positions))

            # Update previous food positions to food positions of current state
            self.previous_food_positions = current_food_positions

            my_pos = game_state.get_agent_position(self.index)


            if missing_food_positions:
                self.target_food = missing_food_positions[0] # Look onto the first missing food if we don't have a target

            #construeer pad tot opgegeten voedsel
            path_to_food = False
            if self.target_food:
                path_to_food = self.breadthFirstSearch(
                    BFSSearchProblem(game_state, my_pos, self.target_food)
                )
            #pad gevonden tot opgegeten voedsel en opportuniteit om enemy te doden
            if path_to_food and scared_timer == 0:
                chosen_action = path_to_food[0]  # First move in BFS path
                lenght_path = len(path_to_food)

                # Check if chosen action is valid
                legalActions = game_state.get_legal_actions(self.index)
                if chosen_action not in legalActions:
                    return random.choice(legalActions)  # Pick a safe fallback
                #enemy is zichtbaar
                if enemy_distances and lenght_path <= 5:
                    min_enemy_distance = min(enemy_distances)
                    score += 10000 / (min_enemy_distance + 1)  # Hoge beloning om dicht bij enemy te zijn
                    if min_enemy_distance <= 3:
                        self.target_food = None #Target food op none, zodat hij toch naar enemy zou gaan ipv pad afmaken
                        score += 100000 / (min_enemy_distance + 1)
                    if my_pos == enemy_positions:
                        score += 1000000 # enemy is dood
                else:
                    return chosen_action

            else: #niks opgegeten of scared_timer > 0
                for enemy in enemy_state:
                    if not missing_food_positions and not enemy.is_pacman and not path_to_food:
                        # zien de enemy niet, niks opgegeten
                        # laat de agent patrouilleren
                        self.patrolling.add(game_state.get_agent_state(self.index))

                        #positie van de teammate
                        teamMatePos = None
                        if game_state.is_on_red_team(self.index):
                            teamMatePos = game_state.get_agent_position(0)
                        else:
                            teamMatePos = game_state.get_agent_position(1)

                        patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]

                        if patrol_dists:
                            min_patrol_dist = min(patrol_dists)
                            score += 1000 / (min_patrol_dist + 1)  # Weak patrol incentive

                        if teamMatePos in patrol_positions:
                            # strategically keep distance from each other
                            distanceToTeamMate = self.get_maze_distance(next_pos, teamMatePos)
                            if distanceToTeamMate <= 7:
                                score -= 2000 / (distanceToTeamMate + 1)

                    else:
                        if game_state.get_agent_state(self.index) in self.patrolling:
                            # stop met patrouilleren
                            self.patrolling.remove(game_state.get_agent_state(self.index))


            # we're scared of our enemy
            if scared_timer > 0:
                for enemi in enemy_state:
                    if enemy_distances and enemi.is_pacman:
                        min_enemy_distance = min(enemy_distances)
                        score -= 1000 / (min_enemy_distance + 1)  # Blijf weg van een aanvallende enemy
                        if next_pos == self.start:
                            score -= 1000000 # agent is dood (terug op start positie)

                        # Blijf op een korte ,veilige afstand van de enemy
                        if 3 <= min_enemy_distance <= 5:
                            score += 100000

            # we're scared but can't see the enemy, so patrol because we can do nothing more
            if scared_timer > 0 and not enemy_distances:
                patrol_dists = [self.get_maze_distance(next_pos, pos) for pos in patrol_positions]
                self.target_food = None
                if patrol_dists:
                    min_patrol_dist = min(patrol_dists)
                    score += 1000 / (min_patrol_dist + 1)  #Dichtsbijzijnde patol position


            # Choose the action with the highest score
            if score > best_score:
                best_score = score
                best_action = action

        #Random actie als fallback
        if best_action is None:
            return random.choice(legalActions)

        return best_action