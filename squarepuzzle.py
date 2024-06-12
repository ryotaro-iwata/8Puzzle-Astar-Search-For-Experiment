from random import randint
from heapq import heappop, heappush
import csv

class SquarePuzzle:
    def __init__(self, edge_length=3, board=None):
        """
        edge_length:一辺の長さ.
        board:初期状態を指定する場合に使う. マスの配置を一次元化したもの.
        """
        if board is not None:
            assert len(board) == edge_length**2, f"invalid square. edge_length={edge_length} and board={board}"
            self.space = [x for x in range(edge_length ** 2) if board[x] == 0][0]
            board = list(board)
        else:
            board = [i + 1 for i in range(edge_length**2)]
            board[-1] = 0  
            self.space = edge_length ** 2 - 1
        self.edge_length = edge_length
        self.board = board
        self.actions = [[0, 1], [0, -1], [1, 0], [-1, 0]]
        self.step_count = 0

    def reset(self, shuffle_count=100):
        
        self.board = [i + 1 for i in range(self.edge_length ** 2)]
        self.board[-1] = 0  
        self.space = self.edge_length ** 2 - 1
        self.step_count = 0
        pre_space = -1
        for _ in range(shuffle_count):
            i, j = divmod(self.space, self.edge_length)
            di, dj = self.actions[randint(0, len(self.actions) - 1)]
            ni, nj = i + di, j + dj
            if 0 <= ni < self.edge_length and 0 <= nj < self.edge_length and ni * self.edge_length + nj != pre_space:
                self.board[self.space], self.board[ni * self.edge_length + nj] = self.board[ni * self.edge_length + nj], self.board[self.space]
                pre_space = self.space
                self.space = ni * self.edge_length + nj
        return tuple(self.board)

    def step(self, action, air=False):

        if not air:
            self.step_count += 1
        i, j = divmod(self.space, self.edge_length)
        di, dj = self.actions[action]
        ni, nj = i + di, j + dj
        if air:
            board_ = self.board.copy()
        else:
            board_ = self.board
        if 0 <= ni < self.edge_length and 0 <= nj < self.edge_length:
            board_[self.space], board_[ni * self.edge_length + nj] = board_[ni * self.edge_length + nj], board_[self.space]
            if not air:
                self.space = ni * self.edge_length + nj
        done = all(board_[i] == (i + 1) % (self.edge_length ** 2) for i in range(self.edge_length ** 2))
        reward = 1 if done else 0
        info = {"step_count": self.step_count}
        return tuple(board_), reward, done, info

    def get_state(self):

        return tuple(self.board)
        
    def get_able_actions(self):

        ret = []
        i, j = divmod(self.space, self.edge_length)
        if j < self.edge_length - 1:
            ret.append(0) # 右
        if 0 < j:
            ret.append(1) # 左
        if i < self.edge_length - 1:
            ret.append(2) # 下
        if 0 < i:
            ret.append(3) # 上
        return ret

    def show(self):

        for i in range(self.edge_length):
            print(self.board[i * self.edge_length:(i + 1) * self.edge_length])
            
class Node0():

    def __init__(self, board, pre=None, action=None):

        self.board = board
        self.edge_length = int(len(board) ** 0.5)
        self.pre = pre
        self.action = action
        self.cost = pre.cost + 1 if pre is not None else 0  # 初期状態からこのノードまでの実コスト
        self.heuristic = self._get_heuristic()  # boardからゴールまでの推定コスト（ヒューリスティック値）
        self.score = self.heuristic + self.cost
    
    def _get_heuristic(self):

        ret = 0
        return ret

    def __le__(self, other):
        return self.score <= other.score
    
    def __ge__(self, other):
        return self.score >= other.score
    
    def __lt__(self, other):
        return self.score < other.score
    
    def __gt__(self, other):
        return self.score > other.score
    
    def __eq__(self, other):
        return self.score == other.score
    
class Node1():

    def __init__(self, board, pre=None, action=None):

        self.board = board
        self.edge_length = int(len(board) ** 0.5)
        self.pre = pre
        self.action = action
        self.cost = pre.cost + 1 if pre is not None else 0  # 初期状態からこのノードまでの実コスト
        self.heuristic = self._get_heuristic()  # boardからゴールまでの推定コスト（ヒューリスティック値）
        self.score = self.heuristic + self.cost
    
    def _get_heuristic(self):
        
        ret = 0
        for i in range(self.edge_length**2-1):
            if not self.board[i] == i+1 : 
                ret+=1
                
        return ret

    # Node比較用関数
    def __le__(self, other):
        return self.score <= other.score
    
    def __ge__(self, other):
        return self.score >= other.score
    
    def __lt__(self, other):
        return self.score < other.score
    
    def __gt__(self, other):
        return self.score > other.score
    
    def __eq__(self, other):
        return self.score == other.score

class Node2():

    def __init__(self, board, pre=None, action=None):

        self.board = board
        self.edge_length = int(len(board) ** 0.5)
        self.pre = pre
        self.action = action
        self.cost = pre.cost + 1 if pre is not None else 0  # 初期状態からこのノードまでの実コスト
        self.heuristic = self._get_heuristic()  # boardからゴールまでの推定コスト（ヒューリスティック値）
        self.score = self.heuristic + self.cost
    
    def _get_heuristic(self):

        ret = 0
        for i in range(self.edge_length):
            for j in range(self.edge_length):
                t = self.board[i * self.edge_length + j] - 1
                t = len(self.board) - 1 if t == -1 else t
                ti, tj = divmod(t, self.edge_length)
                ret += abs(i - ti) + abs(j - tj)
        return ret

    # Node比較用関数
    def __le__(self, other):
        return self.score <= other.score
    
    def __ge__(self, other):
        return self.score >= other.score
    
    def __lt__(self, other):
        return self.score < other.score
    
    def __gt__(self, other):
        return self.score > other.score
    
    def __eq__(self, other):
        return self.score == other.score

def A_star_solver0(env):
    
    board = env.get_state()
    board_tuple = tuple(board)  # リストをタプルに変換

    # 初めから最終状態のときの処理
    if all(board[i] == (i + 1) % (len(board)) for i in range(len(board))):
        return True, [], 0

    node = Node0(board, pre=None, action=None)
    dist = {}
    dist[board_tuple] = node
    q = []
    heappush(q, node)
    search_count = 0
    end_node = None
    
    while end_node is None and q:
        search_count += 1
        
        node = heappop(q)
        env_ = SquarePuzzle(env.edge_length, node.board)
        
        for action in env_.get_able_actions():
            next_state, reward, done, info = env_.step(action, air=True)
            next_state_tuple = tuple(next_state)  # リストをタプルに変換
            next_node = Node0(next_state, pre=node, action=action)
            
            if done:
                end_node = next_node
                break
            
            if next_state_tuple in dist and dist[next_state_tuple] <= next_node:
                continue
            
            dist[next_state_tuple] = next_node
            heappush(q, next_node)
    
    if end_node is None:
        return False, [], search_count
    
    node = end_node
    actions = []
    
    while node.pre is not None:
        actions.append(node.action)
        node = node.pre
    
    actions.reverse()
    return True, actions, search_count

def A_star_solver1(env):
    
    board = env.get_state()
    board_tuple = tuple(board)  # リストをタプルに変換

    # 初めから最終状態のときの処理
    if all(board[i] == (i + 1) % (len(board)) for i in range(len(board))):
        return True, [], 0

    node = Node1(board, pre=None, action=None)
    dist = {}
    dist[board_tuple] = node
    q = []
    heappush(q, node)
    search_count = 0
    end_node = None
    
    while end_node is None and q:
        search_count += 1
        
        node = heappop(q)
        env_ = SquarePuzzle(env.edge_length, node.board)
        
        for action in env_.get_able_actions():
            next_state, reward, done, info = env_.step(action, air=True)
            next_state_tuple = tuple(next_state)  # リストをタプルに変換
            next_node = Node1(next_state, pre=node, action=action)
            
            if done:
                end_node = next_node
                break
            
            if next_state_tuple in dist and dist[next_state_tuple] <= next_node:
                continue
            
            dist[next_state_tuple] = next_node
            heappush(q, next_node)
    
    if end_node is None:
        return False, [], search_count
    
    node = end_node
    actions = []
    
    while node.pre is not None:
        actions.append(node.action)
        node = node.pre
    
    actions.reverse()
    return True, actions, search_count

def A_star_solver2(env):

    board = env.get_state()
    board_tuple = tuple(board)  # リストをタプルに変換

    # 初めから最終状態のときの処理
    if all(board[i] == (i + 1) % (len(board)) for i in range(len(board))):
        return True, [], 0

    node = Node2(board, pre=None, action=None)
    dist = {}
    dist[board_tuple] = node
    q = []
    heappush(q, node)
    search_count = 0
    end_node = None
    
    while end_node is None and q:
        search_count += 1
        
        node = heappop(q)
        env_ = SquarePuzzle(env.edge_length, node.board)
        
        for action in env_.get_able_actions():
            next_state, reward, done, info = env_.step(action, air=True)
            next_state_tuple = tuple(next_state)  # リストをタプルに変換
            next_node = Node2(next_state, pre=node, action=action)
            
            if done:
                end_node = next_node
                break
            
            if next_state_tuple in dist and dist[next_state_tuple] <= next_node:
                continue
            
            dist[next_state_tuple] = next_node
            heappush(q, next_node)

    
    if end_node is None:
        return False, [], search_count
    
    node = end_node
    actions = []
    
    while node.pre is not None:
        actions.append(node.action)
        node = node.pre
    
    actions.reverse()
    return True, actions, search_count

def IDDFS_solver(env,logger=None):

    def log(message):
        if logger is not None:
            logger.info(message)

    board = env.get_state()
    # はじめから最終状態のときの処理
    if all(board[i]==(i+1)%(len(board)) for i in range(len(board))):
        return True,[],0

    log("search start.")
    th = 0
    search_count = 0
    end_node = None
    while end_node is None and th <= 31:
        th += 1
        board = env.get_state()
        node = Node0(board,pre=None,action=None)
        stc = []
        stc.append(node)
        while end_node is None and stc:
            search_count += 1
            if search_count % 1000 == 0:log(f"..search count = {search_count}. th = {th}.")
            node = stc.pop()
            env_ = SquarePuzzle(env.edge_length,node.board)
            for action in env_.get_able_actions():
                next_state, reward, done, info = env_.step(action,air=True)
                next_node = Node0(next_state,pre=node,action=action)
                if done:
                    end_node = next_node
                    break
                if node.pre is not None and next_state == node.pre.board:continue
                if next_node.cost > th:continue
                stc.append(next_node)

    log("search end.")
    if end_node is None:
        log("no answer.")
        return False,[],search_count
    node = end_node
    actions = []
    while node.pre is not None:
        actions.append(node.action)
        node = node.pre
    actions.reverse()
    return True,actions,search_count


def main():
    edge_length = 3  # 一辺の長さ (3なら8パズル、4なら15パズル)
    env = SquarePuzzle(edge_length)

    results = []

    for i in range(10):
        initial_board = env.reset()
        
        env0 = SquarePuzzle(edge_length, initial_board)
        success0, actions0, search_count0 = A_star_solver0(env0)
        
        env1 = SquarePuzzle(edge_length, initial_board)
        success1, actions1, search_count1 = A_star_solver1(env1)
        
        env2 = SquarePuzzle(edge_length, initial_board)
        success2, actions2, search_count2 = A_star_solver2(env2)
        
        env3 = SquarePuzzle(edge_length, initial_board)
        success3, actions3, search_count3 = IDDFS_solver(env3)
        
        results.append({
            'initial_board': initial_board,
            'h0_success': success0,
            'h0_moves': len(actions0) if success0 else -1,
            'h0_search_count': search_count0,
            
            'h1_success': success1,
            'h1_moves': len(actions1) if success1 else -1,
            'h1_search_count': search_count1,
            
            'h2_success': success2,
            'h2_moves': len(actions2) if success2 else -1,
            'h2_search_count': search_count2,
            
            'ids_success': success3,
            'ids_moves': len(actions3) if success3 else -1,
            'ids_search_count': search_count3            
        })

    # CSVファイルに結果を書き出す
    with open('results.csv', 'w', newline='') as csvfile:
        fieldnames = ['initial_board', 'h0_success', 'h0_moves', 'h0_search_count', 'h1_success', 'h1_moves', 'h1_search_count', 'h2_success', 'h2_moves', 'h2_search_count','ids_success','ids_moves','ids_search_count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in results:
            writer.writerow(result)

if __name__ == "__main__":
    main()
    
