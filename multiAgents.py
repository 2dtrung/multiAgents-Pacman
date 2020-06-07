# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.
      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        curr_food = currentGameState.getFood()
        evaluationNumber = 0.0

        for food in curr_food.asList():
            foodDistance = manhattanDistance(newPos, food)
            if foodDistance != 0:
                evaluationNumber += 1.0 / foodDistance
            else:
                evaluationNumber += 2

        for ghost in newGhostStates:
            ghostDistance = manhattanDistance(newPos, ghost.getPosition())
            if ghostDistance <= 3:
                evaluationNumber -= 3

        return evaluationNumber

        

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.
      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.
      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.
          Here are some method calls that might be useful when implementing minimax.
          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1
          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action
          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"

        return self.maxValue(gameState, self.depth)[1]

        def maxValue(self, gameState, depth):
            if (gameState.isWin() or gameState.isLose() or depth == 0):
                return self.evaluationFunction(gameState), ""

            scores = []
            actions = gameState.getLegalActions()
            for a in actions:
                scores.append(self.minValue(gameState.generateSuccessor(self.index, a), depth, 1))
            maxScore = max(scores)
            act = scores.index(maxScore)

            return maxScore, actions[act]

        def minValue(self, gameState, depth, agentIndex):
            if (gameState.isWin() or gameState.isLose() or depth == 0):
                return self.evaluationFunction(gameState), ""

            scores = []
            actions = gameState.getLegalActions(agentIndex)
            for a in actions:
                if(agentIndex == gameState.getNumAgents() - 1):
                    scores.append(self.maxValue(gameState.generateSuccessor(agentIndex, a), (depth-1))[0])
                else:
                    scores.append(self.minValue(gameState.generateSuccessor(agentIndex, a), depth, (agentIndex+1))[0])
            minScore = min(scores)
            act = scores.index(minScore)

            return minScore, actions[act]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minMaxValue(gameState, depth, agent, x, y):
            if agent >= gameState.getNumAgents():
                agent = 0
                depth += 1
            if (gameState.isWin() or gameState.isLose() or depth == self.depth):
                return self.evaluationFunction(gameState)
            elif (agent == 0):
                return maxValue(gameState, depth, agent, x, y)
            else:
                return minValue(gameState, depth, agent, x, y)
        
        def maxValue(gameState, depth, agent, x, y):
            pacmanActions = gameState.getLegalActions(agent)
            output = ["ok", -float("inf")]
            
            if not pacmanActions:
                return self.evaluationFunction(gameState)
                
            for action in pacmanActions:
                curr_state = gameState.generateSuccessor(agent, action)
                curr_value = minMaxValue(curr_state, depth, agent+1, x, y)
                
                if type(curr_value) is list:
                    test_value = curr_value[1]
                else:
                    test_value = curr_value
                    
                if test_value > output[1]:
                    output = [action, test_value]
                if test_value > y:
                    return [action, test_value]
                x = max(x, test_value)
            return output
            
        def minValue(gameState, depth, agent, x, y):
            ghostActions = gameState.getLegalActions(agent)
            output = ["ok", float("inf")]
           
            if not ghostActions:
                return self.evaluationFunction(gameState)
                
            for action in ghostActions:
                curr_state = gameState.generateSuccessor(agent, action)
                curr_value = minMaxValue(curr_state, depth, agent+1, x, y)
                
                if type(curr_value) is list:
                    test_value = curr_value[1]
                else:
                    test_value = curr_value
                    
                    
                if test_value < output[1]:
                    output = [action, test_value]
                if test_value < x:
                    return [action, test_value]
                y = min(y, test_value)
            return output
             
        output_list = minMaxValue(gameState, 0, 0, -float("inf"), float("inf"))
        return output_list[0]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction
          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        action = None
        # if terminal node
        if self.depth == 0:
            return self.evaluationFunction(gameState)
        # if x is max node
        if self.index == 0:
            actions = gameState.getLegalActions(self.index)
            value = float("inf")
            val = []
            for x in actions:
                val.append((self.expValue(gameState.generateSuccessor(self.index, x), 0, 1), x))
            value, action = max(val)

        return action

    def maxValue(self, gameState, depth, agentIndex):
        value = float("-inf")
        actions = gameState.getLegalActions(agentIndex)
        val = []
        if (len(actions) == 0 or depth == self.depth):
            return self.evaluationFunction(gameState)
        else:
            for x in actions:
                val.append(self.expValue(gameState.generateSuccessor(agentIndex, x), depth, (agentIndex + 1)))
            # return max(values)
            value = max(val)
        return value


    def expValue(self, gameState, depth, agentIndex):
        val = []
        actions = gameState.getLegalActions(agentIndex)
        if (len(actions) == 0 or depth == self.depth):
            return self.evaluationFunction(gameState)
        else:
            for x in actions:
                if agentIndex != gameState.getNumAgents() - 1:
                    val.append(self.expValue(gameState.generateSuccessor(agentIndex, x), depth, (agentIndex + 1)))
                else:
                    val.append(self.maxValue(gameState.generateSuccessor(agentIndex, x), (depth + 1), 0))
            # weight = [probability (s,s') for s' in successors(s)]
            # take the average
            weight = sum(val) / len(val)
        return weight

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).
      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    evaluationNumber = 0.0
    # find mininum distance of food in this state
    curr_food = currentGameState.getFood()
    curr_pos = currentGameState.getPacmanPosition()

    mdist = float("inf")
    for food in curr_food.asList():
        foodDistance = manhattanDistance(curr_pos, food)
        if foodDistance < mdist:
            mdist = foodDistance

    # find if there are close ghosts
    ghostPositions = currentGameState.getGhostPositions()

    ghost_score = 0
    for ghost in ghostPositions:
        ghostDistance = util.manhattanDistance(curr_pos, ghost)
        if ghostDistance < 2:
            ghost_score = float("inf")

    # return evalNum
    if mdist != float("inf"):
        evaluationNumber = - mdist
    evaluationNumber += (-ghost_score)- (1000*len(curr_food.asList())) + (10*currentGameState.getScore()) - (10*len(currentGameState.getCapsules()))

    return evaluationNumber

# Abbreviation
better = betterEvaluationFunction