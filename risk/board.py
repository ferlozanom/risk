import os
import random
import heapdict
from collections import namedtuple
from collections import deque
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

import risk.definitions

Territory = namedtuple('Territory', ['territory_id', 'player_id', 'armies'])
Move = namedtuple('Attack', ['from_territory_id', 'from_armies', 'to_territory_id', 'to_player_id', 'to_armies'])
#player_id = color circle
#armies = number inside circle 

#we will probably only need 50 lines of code, many of this are not necessary
class Board(object):
    """
    The Board object keeps track of all armies situated on the Risk
    world map. Through the definitions it knows the locations of and
    connections between all territories. It handles ownership, attacks
    and movements of armies.

    Args:
        data (list): a sorted list of tuples describing the state of the
            board, each containing three values:
            - tid (int): the territory id of a territory,
            - pid (int): the player id of the owner of the territory,
            - n_armies (int): the number of armies on the territory.
            The list is sorted by the tid, and should be complete.
    """

    def __init__(self, data):
        self.data = data

    @classmethod #alternative constructor 
    def create(cls, n_players):
        """
        Create a Board and randomly allocate the territories. Place one army on each territory.
        
        Args:
            n_players (int): Number of players.
                
        Returns:
            Board: A board with territories randomly allocated to the players.
        """
        allocation = (list(range(n_players)) * 42)[0:42]
        random.shuffle(allocation)
        return cls([Territory(territory_id=tid, player_id=pid, armies=1) for tid, pid in enumerate(allocation)])

    # ====================== #
    # == Neighbor Methods == #
    # ====================== #   

    #when you are doing breath first search, you will need to know what are the neighboors of a particular territory, important function
    def neighbors(self, territory_id):
        """
        Create a generator of all territories neighboring a given territory.
            
        Args:
            territory_id (int): ID of the territory to find neighbors of.

        Returns:
            generator: Generator of Territories.
        """
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if t.territory_id in neighbor_ids)

#will also be useful, filter out enemy and friendly neighboors, for two functions where you are creating paths through enemies that you can attack
    def hostile_neighbors(self, territory_id):
        """
        Create a generator of all territories neighboring a given territory, of which
        the owner is not the same as the owner of the original territory.
            
        Args:
            territory_id (int): ID of the territory.
                
        Returns:
            generator: Generator of Territories.
        """
        player_id = self.owner(territory_id)
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if (t.player_id != player_id and t.territory_id in neighbor_ids))

#will be useful for fortification move, take units from one to another territory that you already own. we don't have to use them directly, we can look at the inside of the code (neighbor_ids)
    def friendly_neighbors(self, territory_id):
        """
        Create a generator of all territories neighboring a given territory, of which
        the owner is the same as the owner of the original territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            generator: Generator of tuples of the form (territory_id, player_id, armies).
        """
        player_id = self.owner(territory_id)
        neighbor_ids = risk.definitions.territory_neighbors[territory_id]
        return (t for t in self.data if (t.player_id == player_id and t.territory_id in neighbor_ids))

    
    # ================== #
    # == Path Methods == #
    # ================== #
#start from up to bottom

    def is_valid_path(self, path):
        '''
        A path is list of territories satisfying two conditions:
        1. For all territories V in the list (except the last one), the next territory W is in the neighbors if V.
        2. No territory is repeated multiple times.
        Valid paths can be of any length (including 0 and 1).

        Args:
            path ([int]): a list of territory_ids which represent the path

        Returns:
            bool: True if the input path is valid
        '''

        visited = []

        if path is None: 
            return True
        for idx,V in enumerate(path):
            #checks if territory is repeated
            if V in visited: 
                return False
            try:
                if path[idx + 1]:
                    # W is next territory
                    W = path[idx + 1]
                    #checks if next territory (W) is not a neighbor of V
                    neighbor_ids = risk.definitions.territory_neighbors[V]
                    if W not in neighbor_ids:
                        return False
            except IndexError:
                pass
            visited.append(V)
        return True
        
    def is_valid_attack_path(self, path):
        '''
        The rules of Risk state that when attacking, 
        a player's armies cannot move through territories they already occupy;
        they must move through enemy territories.
        All valid attacks, therefore, will follow a path of starting on one player's territory and moving trough enemy territories.

        Formally, an attack path is a valid path satisfying the following two additional properties:
        1. An attack path must contain at least two territories;
        #all of the remaining nodes in the paths cannot have the same owner as the very first node in the path
        1. If the first territory is owned by player A, then no other territories in the path are also owned by A.

        Args:
            path ([int]): a list of territory_ids which represent the path

        Returns:
            bool: True if the path is an attack path
        '''
        visited = []

        if len(path) < 2: 
            return False

        player_id = self.owner(path[0])

        for idx,V in enumerate(path):
            #checks if territory is repeated
            if V in visited: 
                return False
            try:
                if path[idx + 1]:
                    # W is next territory
                    W = path[idx + 1]
                    neighbor_ids = risk.definitions.territory_neighbors[V]
                    neighbor_owner = self.owner(W)
                    #checks if next territory (W) is not a hostile neighbor of V
                    if W not in neighbor_ids or player_id == neighbor_owner:
                        return False
            except IndexError:
                pass
            visited.append(V)
        return True

    def cost_of_attack_path(self, path):
        '''
        The cost of an attack path is the total number of enemy armies in the path.
        In other words, it is the total number of armies in the subpath starting at the second vertex.

        Args:
            path ([int]): a list of territory_ids which must be a valid attack path

        Returns:
            bool: the number of enemy armies in the path
        '''

        t_armies = 0
        if path is None: 
            return 0
        for idx in path:
            if idx == path[0]:
                pass
            else: 
                armies_in_territory = self.armies(idx)
                t_armies += armies_in_territory

        return t_armies 

    def shortest_path(self, source, target):
        '''
        This function uses BFS to find the shortest path between source and target.
        This function does not take into account who owns the territories or how many armies are on the territories,
        and so a shortest path is simply a valid path with the smallest number of territories visited.
        This path is not necessarily unique,
        and when multiple shortest paths exist,
        then this function can return any of those paths.

        Args:
            source (int): a territory_id that is the source location
            target (int): a territory_id that is the target location

        Returns:
            [int]: a valid path between source and target that has minimum length; this path is guaranteed to exist
        '''
        #Create a dictionary whose keys are territories and values are path
        Dict = {}
        #Set dictionary[source] = [source]
        Dict[source] = [source]
        #Create a queue
        q = deque()
        #Enqueue source onto the queue
        q.append(source)
        #Create a set of visited territories
        visited_t = set()
        #Add source to the set
        visited_t.add(source) 
        #While the queue is not empty
        while q:
            #Dequeue current_territory from the queue
            current_t = q.popleft()
            #If current_territory is the target
            if current_t == target:
                #return the dictionary[current_territory]
                return Dict[current_t] 
            #For each territory in the neighbors of current_territory that is not in the visited set
            neighbor_ids = risk.definitions.territory_neighbors[current_t]
            for t in neighbor_ids:
                if t not in visited_t: 
                    #Make a copy of dictionary[current_territory]
                    Dict_copy = deepcopy(Dict[current_t])
                    #Push territory onto the copy
                    Dict_copy.append(t)
                    #Set dictionary[territory] = copy + territory
                    Dict[t] = Dict_copy
                    #Enqueue territory
                    q.append(t)
                #Add territory to the visited set
                visited_t.add(t)
        

    def can_fortify(self, source, target):
        '''
        At the end of a turn, a player may choose to fortify a target territory by moving armies from a source territory.
        In order for this to be a valid move,
        there must be a valid path between the source and target territories that is owned entirely by the same player.

        Args:
            source (int): the source territory_id
            target (int): the target territory_id

        Returns:
            bool: True if reinforcing the target from the source territory is a valid move
        '''
        valid_move = False
        player_id = self.owner(source)

        #Create a dictionary whose keys are territories and values are path
        Dict = {}
        #Set dictionary[source] = [source]
        Dict[source] = [source]
        #Create a queue
        q = deque()
        #Enqueue source onto the queue
        q.append(source)
        #Create a set of visited territories
        visited_t = set()
        #Add source to the set
        visited_t.add(source)
        #While the queue is not empty
        while q:
            #Dequeue current_territory from the queue
            current_t = q.popleft()
            #If current_territory is the target
            if current_t == target:
                #return the dictionary[current_territory]
                valid_move = True
            #For each territory in the neighbors of current_territory that is not in the visited set
            neighbor_ids = risk.definitions.territory_neighbors[current_t]
            for t in neighbor_ids:
                neighbor_id = self.owner(t)
                if t not in visited_t and neighbor_id == player_id:
                    #Make a copy of dictionary[current_territory]
                    Dict_copy = deepcopy(Dict[current_t])
                    #Push territory onto the copy
                    Dict_copy.append(t)
                    #Set dictionary[territory] = copy + territory
                    Dict[t] = Dict_copy
                    #Enqueue territory
                    q.append(t)
                #Add territory to the visited set
                visited_t.add(t)

        return valid_move

#first calculate what the path is and then if you get a path return true if you dont then false
#neighbor adjacent + also the same color

    def cheapest_attack_path(self, source, target):
        '''
        This function uses Dijkstra's algorithm to calculate a cheapest valid attack path between two territories if such a path exists.
        There may be multiple valid cheapest attack paths (in which case it doesn't matter which this function returns),
        or there may be no valid attack paths (in which case the function returns None).

        Args:
            source (int): territory_id of source node
            target (int): territory_id of target node

        Returns:
            [int]: a list of territory_ids representing the valid attack path; if no path exists, then it returns None instead

        ####################
        Create a dictionary whose keys are territories and values are path
        Set dictionary[source] = [source]
    ++  Create a PRIORITY queue
    ++  Enqueue source onto the PRIORITY queue WITH PRIORITY 0
        Create a set of visited territories
        Add source to the set

    ++  While the PRIORITY queue is not empty
    ++      Dequeue current_territory from the PRIORITY queue
            If current_territory is the target
                return the dictionary[current_territory]
            For each territory in the neighbors of current_territory that is not in the visited set
                Make a copy of dictionary[current_territory]
                Push territory onto the copy
    ++          CALCULATE THE PRIORITY OF THE PATH AS PRIORITY OF CURRENT_TERRITORY + NUMBER OF ARMIES ON TERRITORY
    ++          IF TERRITORY NOT IN THE PRIORITY QUEUE
                    Set dictionary[current_territory] = copy + territory
    ++              Enqueue territory WITH PRIORITY
    ++          ELSE, IF THE NEW PRIORITY IS LESS THEN THE PRIORITY IN THE QUEUE
                    Set dictionary[current_territory] = copy + territory
    ++              UPDATE THE TERRITORY'S PRIORITY IN THE PRIORITY QUEUE WITH THE NEW PRIORITY
            Add current_territory to the visited set

        ###########################

        hd = heapdict.heapdict()
        hd['argentina'] = 5
        hd['brazil'] = 4

        hd.popitem() #returns [k,v], get minimum value in a dictionary at a given moment k = key, v = value 

        the values inside priority queue are going to be the total number of armies in a given territory and we always want to be looking at what is the next territory that has the lowes number of armies 


        '''
        player_id = self.owner(source)

        #Create a dictionary whose keys are territories and values are path
        Dict = {}
        #Set dictionary[source] = [source]
        Dict[source] = [source]
        #Create a queue
        hd = heapdict.heapdict()
        #Enqueue source onto the queue
        hd[source] = 0
        #Create a set of visited territories
        visited_t = set()
        #Add source to the set
        visited_t.add(source)
        #While the queue is not empty
        while hd:
            #Dequeue current_territory from the queue
            (current_t,priority) = hd.popitem()
            #If current_territory is the target
            if current_t == target:
                if current_t == source:
                    return None
                #return the dictionary[current_territory]
                return Dict[current_t]
            #For each territory in the neighbors of current_territory that is not in the visited set
            neighbor_ids = risk.definitions.territory_neighbors[current_t]
            for t in neighbor_ids:
                neighbor_owner = self.owner(t)
                if t not in visited_t and neighbor_owner != player_id:
                    #Make a copy of dictionary[current_territory]
                    Dict_copy = deepcopy(Dict[current_t])
                    #Push territory onto the copy
                    Dict_copy.append(t)
                    priority_path = priority + self.armies(t)
                    # how can you check if t in hd? 
                    if t not in hd:
                        #Set dictionary[territory] = copy + territory
                        Dict[t] = Dict_copy
                        print('Dict[t]=',Dict[t])
                        #Enqueue territory
                        hd[t] = priority_path
                        print('path_priority=',priority_path)
                    else:
                        if priority_path < hd[t]:
                            Dict[t] = Dict_copy
                            print('previous priority=',hd[t])
                            hd[t] = priority_path
                            print('new priority=', hd[t])
                #Add territory to the visited set
                visited_t.add(t)

    def can_attack(self, source, target):
        '''
        Args:
            source (int): territory_id of source node
            target (int): territory_id of target node

        Returns:
            bool: True if a valid attack path exists between source and target; else False
        '''
        player_id = self.owner(source)
        valid_move = False  

        #Create a dictionary whose keys are territories and values are path
        Dict = {}
        #Set dictionary[source] = [source]
        Dict[source] = [source]
        #Create a queue
        hd = heapdict.heapdict()
        #Enqueue source onto the queue
        hd[source] = 0
        #Create a set of visited territories
        visited_t = set()
        #Add source to the set
        visited_t.add(source)
        #While the queue is not empty
        while hd:
            #Dequeue current_territory from the queue
            (current_t,priority) = hd.popitem()
            #If current_territory is the target
            if current_t == target:
                if current_t == source:
                    return None
                #return the dictionary[current_territory]
                valid_move = True
            #For each territory in the neighbors of current_territory that is not in the visited set
            neighbor_ids = risk.definitions.territory_neighbors[current_t]
            for t in neighbor_ids:
                neighbor_owner = self.owner(t)
                if t not in visited_t and neighbor_owner != player_id:
                    #Make a copy of dictionary[current_territory]
                    Dict_copy = deepcopy(Dict[current_t])
                    #Push territory onto the copy
                    Dict_copy.append(t)
                    priority_path = priority + self.armies(t)
                    # how can you check if t in hd? 
                    if t not in hd:
                        #Set dictionary[territory] = copy + territory
                        Dict[t] = Dict_copy
                        print('Dict[t]=',Dict[t])
                        #Enqueue territory
                        hd[t] = priority_path
                        print('path_priority=',priority_path)
                    else:
                        if priority_path < hd[t]:
                            Dict[t] = Dict_copy
                            print('previous priority=',hd[t])
                            hd[t] = priority_path
                            print('new priority=', hd[t])
                #Add territory to the visited set
                visited_t.add(t)

        return valid_move

#some of the algorithms that you have to implement are already implemented down here
    # ==================== #
    # == Action Methods == #
    # ==================== #    

    def reinforcements(self, player_id):
        """
        Calculate the number of reinforcements a player is entitled to.
            
        Args:
            player_id (int): ID of the player.

        Returns:
            int: Number of reinforcement armies that the player is entitled to.
        """
        base_reinforcements = max(3, int(self.n_territories(player_id) / 3))
        bonus_reinforcements = 0
        for continent_id, bonus in risk.definitions.continent_bonuses.items():
            if self.continent_owner(continent_id) == player_id:
                bonus_reinforcements += bonus
        return base_reinforcements + bonus_reinforcements

    def possible_attacks(self, player_id):#down here!
        """
        Assemble a list of all possible attacks for the players.

        Args:
            player_id (int): ID of the attacking player.

        Returns:
            list: List of Moves.
        """
        return [Move(from_t.territory_id, from_t.armies, to_t.territory_id, to_t.player_id, to_t.armies)
                for from_t in self.mobile(player_id) for to_t in self.hostile_neighbors(from_t.territory_id)]

    def possible_fortifications(self, player_id):
        """
        Assemble a list of all possible fortifications for the players.
        
        Args:
            player_id (int): ID of the attacking player.

        Returns:
            list: List of Moves.
        """
        return [Move(from_t.territory_id, from_t.armies, to_t.territory_id, to_t.player_id, to_t.armies)
                for from_t in self.mobile(player_id) for to_t in self.friendly_neighbors(from_t.territory_id)]

    def fortify(self, from_territory, to_territory, n_armies):
        """
        Perform a fortification.

        Args:
            from_territory (int): Territory_id of the territory where armies leave.
            to_territory (int): Territory_id of the territory where armies arrive.
            n_armies (int): Number of armies to move.

        Raises:
            ValueError if the player moves too many or negative armies.
            ValueError if the territories do not share a border or are not owned by the same player.
        """
        if n_armies < 0 or self.armies(from_territory) <= n_armies:
            raise ValueError('Board: Cannot move {n} armies from territory {tid}.'
                             .format(n=n_armies, tid=from_territory))
        if to_territory not in [t.territory_id for t in self.friendly_neighbors(from_territory)]:
            raise ValueError('Board: Cannot fortify, territories do not share owner and/or border.')
        self.add_armies(from_territory, -n_armies)
        self.add_armies(to_territory, +n_armies)

    def attack(self, from_territory, to_territory, attackers):
        """
        Perform an attack.

        Args:
            from_territory (int): Territory_id of the offensive territory.
            to_territory (int): Territory_id of the defensive territory.
            attackers (int): Number of attacking armies.

        Raises:
            ValueError if the number of armies is <1 or too large.
            ValueError if a player attacks himself or the territories do not share a border.

        Returns:
            bool: True if the defensive territory was conquered, False otherwise.
        """
        if attackers < 1 or self.armies(from_territory) <= attackers:
            raise ValueError('Board: Cannot attack with {n} armies from territory {tid}.'
                             .format(n=attackers, tid=from_territory))
        if to_territory not in [tid for (tid, _, _) in self.hostile_neighbors(from_territory)]:
            raise ValueError('Board: Cannot attack, territories do not share border or are owned by the same player.')
        defenders = self.armies(to_territory)
        def_wins, att_wins = self.fight(attackers, defenders)
        if self.armies(to_territory) == att_wins:
            self.add_armies(from_territory, -attackers)
            self.set_armies(to_territory, attackers - def_wins)
            self.set_owner(to_territory, self.owner(from_territory))
            return True
        else:
            self.add_armies(from_territory, -def_wins)
            self.add_armies(to_territory, -att_wins)
            return False

    # ====================== #
    # == Plotting Methods == #
    # ====================== #    
#will be very useful for debugging, especially board.plot_board() and board.plot_board(path = path) 
    def plot_board(self, path=None, plot_graph=False, filename=None):
        """ 
        Plot the board. 
        
        Args:
            path ([int]): a path of territory_ids to plot
            plot_graph (bool): if true, plots the graph structure overlayed on the board
            filename (str): if given, the plot will be saved to the given filename instead of displayed
        """
        im = plt.imread(os.getcwd() + '/img/risk.png')
        dpi=96
        img_width=800
        fig, ax = plt.subplots(figsize=(img_width/dpi, 300/dpi), dpi=dpi)
        _ = plt.imshow(im)
        plt.axis('off')
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())

        def plot_path(xs):
            if not self.is_valid_path(xs):
                print('WARNING: not a valid path')
            coor = risk.definitions.territory_locations[xs[0]]
            verts=[(coor[0]*1.2, coor[1]*1.22 + 25)]
            codes = [ Path.MOVETO ]
            for i,x in enumerate(xs[1:]):
                if (xs[i]==19 and xs[i+1]==1) or (xs[i]==1 and xs[i+1]==19):
                    coor = risk.definitions.territory_locations[x]
                    #verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    verts.append((1000,-200))
                    verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    codes.append(Path.CURVE3)
                    codes.append(Path.CURVE3)
                else:
                    coor = risk.definitions.territory_locations[x]
                    verts.append((coor[0]*1.2, coor[1]*1.22 + 25))
                    codes.append(Path.LINETO)
            path = Path(verts, codes)
            patch = patches.PathPatch(path, facecolor='none', lw=2)
            ax.add_patch(patch)

        if path is not None:
            plot_path(path)

        if plot_graph:
            for t in risk.definitions.territory_neighbors:
                path = []
                for n in risk.definitions.territory_neighbors[t]:
                    path.append(t)
                    path.append(n)
                plot_path(path)

        for t in self.data:
            self.plot_single(t.territory_id, t.player_id, t.armies)

        if not filename:
            plt.show()
        else:
            plt.tight_layout()
            plt.savefig(filename,bbox_inches='tight')

    @staticmethod
    def plot_single(territory_id, player_id, armies):
        """
        Plot a single army dot.
            
        Args:
            territory_id (int): the id of the territory to plot,
            player_id (int): the player id of the owner,
            armies (int): the number of armies.
        """
        coor = risk.definitions.territory_locations[territory_id]
        plt.scatter(
            [coor[0]*1.2], 
            [coor[1]*1.22], 
            s=100, 
            c=risk.definitions.player_colors[player_id],
            zorder=2
            )
        plt.text(
            coor[0]*1.2, 
            coor[1]*1.22 + 25, 
            s=str(armies),
            color='black' if risk.definitions.player_colors[player_id] in ['yellow', 'pink'] else 'white',
            ha='center', 
            size=7
            )
#this will be useful, when its talking about a player it is stored as int [0,1,2..]. When we are deciding if it is okay to attack for x to z we have to make sure the owners are different.. 
    # ======================= #
    # == Territory Methods == #
    # ======================= #

    def owner(self, territory_id):
        """
        Get the owner of the territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            int: Player_id that owns the territory.
        """
        return self.data[territory_id].player_id
#when trying to determine the shortest path that you have to take, this function will be useful to determine what is the weight on each of the nodes, the number of armies on each of the nodes is the weight when we are determining what we call an attackmap

    def armies(self, territory_id):
        """
        Get the number of armies on the territory.

        Args:
            territory_id (int): ID of the territory.

        Returns:
            int: Number of armies in the territory.
        """
        return self.data[territory_id].armies

    def set_owner(self, territory_id, player_id):
        """
        Set the owner of the territory.

        Args:
            territory_id (int): ID of the territory.
            player_id (int): ID of the player.
        """
        self.data[territory_id] = Territory(territory_id, player_id, self.armies(territory_id))

    def set_armies(self, territory_id, n):
        """
        Set the number of armies on the territory.

        Args:
            territory_id (int): ID of the territory.
            n (int): Number of armies on the territory.

        Raises:
            ValueError if n < 1.
        """
        if n < 1:
            raise ValueError('Board: cannot set the number of armies to <1 ({tid}, {n}).'.format(tid=territory_id, n=n))
        self.data[territory_id] = Territory(territory_id, self.owner(territory_id), n)

    def add_armies(self, territory_id, n):
        """
        Add (or remove) armies to/from the territory.

        Args:
            territory_id (int): ID of the territory.
            n (int): Number of armies to add to the territory.

        Raises:
            ValueError if the resulting number of armies is <1.
        """
        self.set_armies(territory_id, self.armies(territory_id) + n)

    def n_armies(self, player_id):
        """
        Count the total number of armies owned by a player.

        Args:
            player_id (int): ID of the player.

        Returns:
            int: Number of armies owned by the player.
        """
        return sum((t.armies for t in self.data if t.player_id == player_id))

    def n_territories(self, player_id):
        """
        Count the total number of territories owned by a player.

        Args:
            player_id (int): ID of the player.

        Returns:
            int: Number of territories owned by the player.
        """
        return len([None for t in self.data if t.player_id == player_id])

    def territories_of(self, player_id):
        """
        Return a set of all territories owned by the player.

        Args:
            player_id (int): ID of the player.

        Returns:
            list: List of all territory IDs owner by the player.
        """
        return [t.territory_id for t in self.data if t.player_id == player_id]

    def mobile(self, player_id):
        """
        Create a generator of all territories of a player which can attack or move,
        i.e. that have more than one army.

        Args:
            player_id (int): ID of the attacking player.

        Returns:
            generator: Generator of Territories.
        """
        return (t for t in self.data if (t.player_id == player_id and t.armies > 1))
