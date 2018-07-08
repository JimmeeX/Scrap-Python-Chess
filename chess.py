import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import re
import math
import itertools
import sys
import random

from copy import deepcopy
from collections import Counter
from matplotlib.widgets import Button

############################## DEFINITIONS ##############################

# Pieces
EMPTY = -1
PAWN = 0
KNIGHT = 1
BISHOP = 2
ROOK = 3
QUEEN = 4
KING = 5

pieceShortName = ["P", "N", "B", "R", "Q", "K"]

pieceNames = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"]

WHITE = 0
BLACK = 1

pieceColours = ["w", "b"]

MOVE = 0
ATTACK = 1

# Normal/Special Moves
NORMAL = 0
CASTLING = 1
EN_PASSANT = 2

HUMAN = 0
COMPUTER = 1

# Mapping Chess Piece to corresponding 64x64 Image file
piecetoImg = {
    (PAWN, WHITE): "Data/wp.png",
    (KNIGHT, WHITE): "Data/wn.png",
    (BISHOP, WHITE): "Data/wb.png",
    (ROOK, WHITE): "Data/wr.png",
    (QUEEN, WHITE): "Data/wq.png",
    (KING, WHITE): "Data/wk.png",
    (PAWN, BLACK): "Data/bp.png",
    (KNIGHT, BLACK): "Data/bn.png",
    (BISHOP, BLACK): "Data/bb.png",
    (ROOK, BLACK): "Data/br.png",
    (QUEEN, BLACK): "Data/bq.png",
    (KING, BLACK): "Data/bk.png"
}

# Abbreviation for Chess Piece (used in extracting Knight from Ne4-f6 for example)
movetoPiece = {
    "P": PAWN,
    "N": KNIGHT,
    "B": BISHOP,
    "R": ROOK,
    "Q": QUEEN,
    "K": KING
}

# Reverse Dictionary for moveToPiece
movetoPiece_rev = {v: k for k, v in movetoPiece.items()}

# Value of each piece
pieceToScore = {
    PAWN: 10,
    KNIGHT: 30,
    BISHOP: 30,
    ROOK: 50,
    QUEEN: 90,
    KING: 900
}

castlingPos = {
    (WHITE, "0-0"): "h1",
    (WHITE, "0-0-0"): "a1",
    (BLACK, "0-0-0"): "a8",
    (BLACK, "0-0"): "h8"
}

castlingPos_rev = {v: k for k, v in castlingPos.items()}

castlingPosPiece = {
    (WHITE, KING, "0-0"): "g1",
    (WHITE, ROOK, "0-0"): "f1",
    (WHITE, KING, "0-0-0"): "c1",
    (WHITE, ROOK, "0-0-0"): "d1",
    (BLACK, KING, "0-0"): "g8",
    (BLACK, ROOK, "0-0"): "f8",
    (BLACK, KING, "0-0-0"): "c8",
    (BLACK, ROOK, "0-0-0"): "d8"
}

# Initialise Coordinate Mapping for Chess Board
ranks = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
files = {"1": 7, "2": 6, "3": 5, "4": 4, "5": 3, "6": 2, "7": 1, "8": 0}

coord_keys = ["".join(item)[::-1] for item in list(itertools.product(files, ranks))]
coord_vals = list(itertools.product(files.values(), ranks.values()))

# Maps coordinates to (row, col) of chess board
coords = dict(zip(coord_keys, coord_vals))

# Reverse Dictionary to coords
coords_rev = {v: k for k, v in coords.items()}

# DataType for chessboard numpy matrix
pieceDtype = [('id', 'i2'), ('colour', 'i2'), ('coord', 'S2')]

# Promotion
promotion = QUEEN

#########################################################################

######################## CLASS DEFINITIONS ##############################

class ChessPiece:
    """
    Representation of Chess Pieces. 

    Attributes:
    id - Kind of piece (eg, PAWN, KNIGHT)
    colour - BLACK or WHITE
    position - String representation of coordinates (eg "a4", "c3") on the chessboard
    """
    def __init__(self, id_, colour_, position_):
        """Constructor"""
        self.id = id_
        self.colour = colour_
        self.position = position_

    def changePosition(self, position_):
        """Modifies coordinates of chess piece"""
        self.position = position_

    def changeID(self, id_):
        """Modifies the identity of chess piece"""
        self.id = id_


class ChessGame:
    """
    Interface for Chess Game

    Attributes:
    chessboard - Stores a list of board positions for every move.
    Eg, chessboard[0] represents the board when the game starts;
        chessboard[1] represents the board after move 1... etc
        chessboard[-1] represents the board after the most recent move

    pieces - A list of type ChessPiece to represent what is on the current chessboard

    moves - A list of strings to represent what moves have been played

    timer - ... TODO
    """
    def __init__(self, timer_):
        self.chessboard = []
        self.piece_history = []
        self.pieces = []
        self.moves = []
        self.timer = timer_
        self.gameStatus = False

    def updateBoard(self):
        """
        Updates current chessboard with the positions given in self.pieces
        """
        board = np.empty((8, 8), dtype=pieceDtype)

        # Initialise every element to empty
        for row in range(8):
            for col in range(8):
                board[row][col] = (EMPTY, -1, "-1")

        # Fill corresponding squares with pieces
        for piece in self.pieces:
            board[coords[piece.position]] = (piece.id, piece.colour, piece.position)

        self.chessboard.append(deepcopy(board))
        self.gameStatus = self.gameOver()
        # print(evaluatePosition(self))
        self.showBoard()

    def resetBoard(self):
        """
        Resets chessboard, pieces, moves
        Initialises positions of the chess pieces
        """
        self.chessboard = []
        self.piece_history = []
        self.pieces = []
        self.moves = []

        # WHITE PIECES
        self.pieces.append(ChessPiece(ROOK, WHITE, "a1"))
        self.pieces.append(ChessPiece(KNIGHT, WHITE, "b1"))
        self.pieces.append(ChessPiece(BISHOP, WHITE, "c1"))
        self.pieces.append(ChessPiece(QUEEN, WHITE, "d1"))
        self.pieces.append(ChessPiece(KING, WHITE, "e1"))
        self.pieces.append(ChessPiece(BISHOP, WHITE, "f1"))
        self.pieces.append(ChessPiece(KNIGHT, WHITE, "g1"))
        self.pieces.append(ChessPiece(ROOK, WHITE, "h1"))
        self.pieces.append(ChessPiece(PAWN, WHITE, "a2"))
        self.pieces.append(ChessPiece(PAWN, WHITE, "b2"))
        self.pieces.append(ChessPiece(PAWN, WHITE, "c2"))
        self.pieces.append(ChessPiece(PAWN, WHITE, "d2"))
        self.pieces.append(ChessPiece(PAWN, WHITE, "e2"))
        self.pieces.append(ChessPiece(PAWN, WHITE, "f2"))
        self.pieces.append(ChessPiece(PAWN, WHITE, "g2"))
        self.pieces.append(ChessPiece(PAWN, WHITE, "h2"))

        # BLACK PIECES
        self.pieces.append(ChessPiece(ROOK, BLACK, "a8"))
        self.pieces.append(ChessPiece(KNIGHT, BLACK, "b8"))
        self.pieces.append(ChessPiece(BISHOP, BLACK, "c8"))
        self.pieces.append(ChessPiece(QUEEN, BLACK, "d8"))
        self.pieces.append(ChessPiece(KING, BLACK, "e8"))
        self.pieces.append(ChessPiece(BISHOP, BLACK, "f8"))
        self.pieces.append(ChessPiece(KNIGHT, BLACK, "g8"))
        self.pieces.append(ChessPiece(ROOK, BLACK, "h8"))
        self.pieces.append(ChessPiece(PAWN, BLACK, "a7"))
        self.pieces.append(ChessPiece(PAWN, BLACK, "b7"))
        self.pieces.append(ChessPiece(PAWN, BLACK, "c7"))
        self.pieces.append(ChessPiece(PAWN, BLACK, "d7"))
        self.pieces.append(ChessPiece(PAWN, BLACK, "e7"))
        self.pieces.append(ChessPiece(PAWN, BLACK, "f7"))
        self.pieces.append(ChessPiece(PAWN, BLACK, "g7"))
        self.pieces.append(ChessPiece(PAWN, BLACK, "h7"))

        self.piece_history.append(deepcopy(self.pieces))
        self.updateBoard()

    def makeMove(self, move):
        """
        Perform a chess move if valid with given move (eg, Ne4-f6)
        """
        # If Game Over
        validMove = 0
        if self.gameStatus is True:
            print("Game is Over, press 'New Game'")
        else:
            if((move == "0-0-0") or (move == "0-0")):
                turn = (len(self.chessboard) - 1) % 2
                # Check Castling is Valid
                if castlingIsLegal(self.chessboard, castlingPos[(turn, move)]):
                    # Move King and Rook
                    for i, piece in enumerate(self.pieces):
                        if(piece.id == KING and piece.colour == turn):
                            self.pieces[i].changePosition(castlingPosPiece[(turn, KING, move)])
                        elif(piece.id == ROOK and piece.colour == turn and piece.position == castlingPos[(turn, move)]):
                            self.pieces[i].changePosition(castlingPosPiece[(turn, ROOK, move)])
                    validMove = 1
                else:
                    print("Illegal Castling")
            else:
                # Formatting and Checking String
                move = move.replace(" ", "").lower()
                if(len(move) == 5):
                    move = "P" + move
                move = move.capitalize()
                pattern = re.compile("([P|N|B|R|Q|K][a-h][1-8][\-|x][a-h][1-8])")
                if(pattern.match(move)):
                    # Get Parameters from String
                    board = self.chessboard[-1]
                    orig = move[1:3]
                    dest = move[4:]
                    piece_id = movetoPiece[move[0]]
                    colour = (len(self.chessboard) - 1) % 2
                    if(move[3] == "-"):
                        mode = MOVE
                    else:
                        mode = ATTACK

                    # Check Move is Valid
                    if moveIsLegal(board, piece_id, colour, mode, orig, dest) or enPassantIsLegal(board, self.moves, piece_id, colour, orig, dest):
                        # If move is enPassant, then change destination to the location of the attacked piece
                        if enPassantIsLegal(board, self.moves, piece_id, colour, orig, dest):
                            move = move[:3] + 'x' + move[4:]
                            if colour == WHITE:
                                temp = dest[0] + str(chr(ord(dest[1]) - 1))
                            else:
                                temp = dest[0] + str(chr(ord(dest[1]) + 1))
                            for i, piece in enumerate(self.pieces):
                                if piece.position == temp:
                                    del self.pieces[i]

                        # If pawn is about to promote...
                        if checkPromotePawn(board, piece_id, colour, orig, dest):
                            move = move[:] + "=" + pieceShortName[promotion]
                            for i, piece in enumerate(self.pieces):
                                if(piece.id == piece_id and piece.colour == colour and piece.position == orig):
                                    self.pieces[i].changeID(promotion)
                                    piece_id = promotion
                                    break

                        # If attack, remove piece
                        if mode == ATTACK:
                            for i, piece in enumerate(self.pieces):
                                if piece.position == dest:
                                    del self.pieces[i]

                        # Perform Move // Update Piece Position
                        for i, piece in enumerate(self.pieces):
                            if(piece.id == piece_id and piece.colour == colour and piece.position == orig):
                                self.pieces[i].changePosition(dest)
                                validMove = 1
                                break

                    else:
                        print("Illegal Move")
                else:
                    print("Invalid Move Format")
            if(validMove):
                self.moves.append(move)
                self.piece_history.append(deepcopy(self.pieces))
                self.updateBoard()

        return validMove

    def gameOver(self):
        """
        Evaluates the position
        1. CHECK (King is attacked)
        2. CHECKMATE (King is attacked and there is no possible defence)
        3. THREEFOLD (same position repeats 3 times in a row)
        4. FIFTY-MOVE RULE (if no pawns have moved or no captures have been made in the last 50 moves)
        """
        board = self.chessboard[-1]
        turn = (len(self.chessboard) - 1) % 2
        if turn == WHITE:
            pro = "WHITE"
            ant = "BLACK"
        else:
            pro = "BLACK"
            ant = "WHITE"
        if checkMate(self.chessboard, turn, self.moves):
            print("CHECKMATE - " + ant + " WINS")
            return True
        elif threefold(self.chessboard):
            print("DRAW - THREEFOLD REPETITION")
            return True
        elif fiftyMoves(self.moves):
            print("DRAW - FIFTY MOVE RULE")
            return True
        elif kingInDanger(board, turn):
            print("CHECK")
        return False

    def getGameStatus(self):
        return self.gameStatus

    def getMoves(self):
        return self.moves

    def getChessboard(self):
        return self.chessboard

    def getPieces(self):
        return self.pieces

    def printMoves(self):
        """
        Prints the history of moves mades
        """
        print("   White    Black")
        for i, move in enumerate(self.moves):
            # White Move
            if not (i % 2):
                print(str(math.ceil((i + 1) / 2)) + ". " + move, end='   ')
            else:
                print(move)

        print("")

    def printScore(self):
        """
        Prints sum of pieces
        """
        whiteScore = 0
        for piece in self.pieces:
            if piece.colour == WHITE and piece.id != KING:
                whiteScore += pieceToScore[piece.id]

        blackScore = 0
        for piece in self.pieces:
            if piece.colour == BLACK and piece.id != KING:
                blackScore += pieceToScore[piece.id]

        print("White: " + str(whiteScore))
        print("Black: " + str(blackScore))
        print("")

    def printPieces(self):
        """
        Prints piece, location
        """
        totalWhite = 0
        totalBlack = 0
        for piece in self.pieces:
            if piece.colour == WHITE:
                totalWhite += 1
            else:
                totalBlack += 1
            print(pieceColours[piece.colour] + ' ' + pieceNames[piece.id] + ' ' + piece.position)
        print("")
        print("Total Pieces: " + str(len(self.pieces)))
        print("White: " + str(totalWhite))
        print("Black: " + str(totalBlack))

    def getTurn(self):
        return (len(self.chessboard) - 1) % 2

    def undoMove(self):
        if len(self.chessboard) > 1:
            # Delete
            del self.chessboard[-1]
            del self.chessboard[-1]
            del self.piece_history[-1]
            del self.moves[-1]
            self.pieces = deepcopy(self.piece_history[-1])

            self.updateBoard()

    def showBoard(self):
        """
        Visualisation of most recent chessboard
        """
        global fig, img

        # Create Empty Board
        white = np.ones((100, 100, 4)) * 0
        black = np.ones((100, 100, 4))
        black[:, :, 0:3] = 128
        black[:, :, 3] = 1
        row1 = np.concatenate([white, black, white, black, white, black, white, black], axis=1)
        row2 = np.concatenate([black, white, black, white, black, white, black, white], axis=1)
        board_vis = np.concatenate([row1, row2, row1, row2, row1, row2, row1, row2], axis=0)
        
        if selection != None:
            # Add Selection Square - Blue
            pos = [100 * x for x in coords[selection]]
            board_vis[pos[0]:pos[0] + 100, pos[1]:pos[1] + 100] = [0, 0, 128, 1]

            board = self.chessboard[-1]
            piece = board[coords[selection]][0]
            colour = board[coords[selection]][1]

            moves = generatePossibleMoves(board, piece, colour, MOVE, selection)
            special_moves = []
            # Special Move - Castling
            if piece == KING and castlingIsLegal(self.chessboard, castlingPos[(colour, "0-0")]):
                special_moves.append(castlingPosPiece[(colour, KING, "0-0")])
            if piece == KING and castlingIsLegal(self.chessboard, castlingPos[(colour, "0-0-0")]):
                special_moves.append(castlingPosPiece[(colour, KING, "0-0-0")])

            # Special Move - En Passant
            enPass_moves = []
            if colour == WHITE and piece == PAWN and selection[1] == '5':
                deltas = [(-1, +1), (-1, -1)]
                row = coords[selection][0]
                col = coords[selection][1]
                enPass_moves = [coords_rev[(row + delta[0], col + delta[1])] for delta in deltas if 0<=row+delta[0]<8 and 0<=col+delta[1]<8]
            elif colour == BLACK and piece == PAWN and selection[1] == '4':
                deltas = [(+1, +1), (+1, -1)]
                row = coords[selection][0]
                col = coords[selection][1]
                enPass_moves = [coords_rev[(row + delta[0], col + delta[1])] for delta in deltas if 0<=row+delta[0]<8 and 0<=col+delta[1]<8]

            attacks = generatePossibleMoves(board, piece, colour, ATTACK, selection)
            pos_moves_list = []
            pos_attacks_list = []
            for move in moves:
                if moveIsLegal(board, piece, colour, MOVE, selection, move):                 
                    pos_moves = [100*x for x in coords[move]]
                    pos_moves_list.append(pos_moves)
            for attack in attacks:
                if moveIsLegal(board, piece, colour, ATTACK, selection, attack):
                    pos_attacks = [100*x for x in coords[attack]]
                    pos_attacks_list.append(pos_attacks)

            for move in special_moves:
                pos_moves = [100*x for x in coords[move]]
                pos_moves_list.append(pos_moves)

            for move in enPass_moves:
                if enPassantIsLegal(board, self.moves, piece, colour, selection, move):
                    pos_moves = [100*x for x in coords[move]]
                    pos_moves_list.append(pos_moves)

            # Add Possible Moves - Green
            for pos in pos_moves_list:
                board_vis[pos[0]:pos[0]+100, pos[1]:pos[1]+100] = [0, 128, 0, 0.5]
            # Add Possible Attacks - Red
            for pos in pos_attacks_list:
                board_vis[pos[0]:pos[0]+100, pos[1]:pos[1]+100] = [128, 0, 0, 0.5]

        # Add Pieces
        for piece in self.pieces:
            piece_arr = plt.imread(piecetoImg[(piece.id, piece.colour)])
            pos = [100*x + 18 for x in coords[piece.position]]
            board_vis[pos[0]:pos[0]+64, pos[1]:pos[1]+64] = piece_arr

        # Show Board
        img.set_data(board_vis)
        fig.canvas.draw()

#########################################################################

######################### FUNCTION DEFINITIONS ##########################
# Helper Functions

# Return 1 if Move is Legal, else return 0
def moveIsLegal(board, piece_id, colour, mode, orig, dest):
    """
    Check is move is legal. DOES not apply to castling & en passant
    """
    # 1. Check if Move is Possible
    # Insert condtition here
    if(moveIsPossible(board, piece_id, colour, mode, orig, dest)):
        # Perform temporary move to check if king will be in danger
        temp = np.copy(board)
        temp[coords[dest]] = temp[coords[orig]]
        temp[coords[orig]] = (EMPTY, -1, "-1")
        
        if not kingInDanger(temp, colour):
            return True
    
    return False

def castlingIsLegal(chessboard, dest):
    """
    Checks if castling is legal for a provided destination
    """
    # White Castling
    if dest[1] == "1":
        orig = "e1"
        if pieceIsStationary(chessboard, orig) and pieceIsStationary(chessboard, dest) and not pieceInBetween(chessboard[-1], orig, dest) and not kingInDanger(chessboard[-1], WHITE):
            # Simulate Moves on a temporary board to see if king would move in a square under attack:
            # While(king is not at destination):
            #   move 1 square towards destination
            #   if(king in danger); return illegal
            squares = squaresInBetween(orig, castlingPosPiece[(WHITE, KING, castlingPos_rev[dest][1])])
            squares.append(castlingPosPiece[(WHITE, KING, castlingPos_rev[dest][1])])
            currBoard = np.copy(chessboard[-1])

            for kingDest in squares:
                # Perform Move
                currBoard[coords[kingDest]] = currBoard[coords[orig]]
                currBoard[coords[orig]] = (EMPTY, -1, "-1")

                # Check Safety
                if kingInDanger(currBoard, WHITE):
                    return False

                # Update Variables
                orig = kingDest
            return True
    # Black Castling
    elif dest[1] == "8":
        orig = "e8"
        if pieceIsStationary(chessboard, orig) and pieceIsStationary(chessboard, dest) and not pieceInBetween(chessboard[-1], orig, dest) and not kingInDanger(chessboard[-1], BLACK):
            squares = squaresInBetween(orig, castlingPosPiece[(BLACK, KING, castlingPos_rev[dest][1])])
            squares.append(castlingPosPiece[(BLACK, KING, castlingPos_rev[dest][1])])
            currBoard = np.copy(chessboard[-1])

            for kingDest in squares:
                # Perform Move
                currBoard[coords[kingDest]] = currBoard[coords[orig]]
                currBoard[coords[orig]] = (EMPTY, -1, "-1")

                # Check Safety
                if kingInDanger(currBoard, BLACK):
                    return False

                # Update Variables
                orig = kingDest
            return True
    return False

def enPassantHelper(moves, colour):
    """
    Check if most recent move is a pawn move: {Black from 7th rank; White from 2nd Rank}
    """
    if len(moves) == 0:
        return 'n'
    lastMove = moves[-1]
    if colour == WHITE:
        # Check if black has moved a pawn from 7th rank to 5th rank
        if lastMove[0] == 'P' and lastMove[2] == "7" and lastMove[-1] == "5":
            return lastMove[1]
    elif colour == BLACK:
        # Check if white has moved a pawn from 2nd rank to 4th rank
        if lastMove[0] == 'P' and lastMove[2] == "2" and lastMove[-1] == "4":
            return lastMove[1]
    # Return 'n' for null
    return 'n'

def enPassantIsLegal(board, moves, piece_id, colour, orig, dest):
    """
    Checks if en passant is legal
    """

    # Requirements
    # Piece to move is pawn
    # Opposing colour has recently moved a pawn two spots
    # Both pawns are on same rank, and are horizontally adjacent to each other
    
    ePLoc = enPassantHelper(moves, colour)
    if colour == WHITE:
        if piece_id == PAWN and ePLoc != 'n' and dest[0] == ePLoc and abs(ord(orig[0]) - ord(ePLoc)) == 1 and ord(dest[1]) - ord(orig[1]) == 1 and orig[1] == '5':
            return True
    elif colour == BLACK:
        if piece_id == PAWN and ePLoc != 'n' and dest[0] == ePLoc and abs(ord(orig[0]) - ord(ePLoc)) == 1 and ord(dest[1]) - ord(orig[1]) == -1 and orig[1] == '4':
            return True
    return False

def checkPromotePawn(board, piece_id, colour, orig, dest):
    """
    Checks if there is any pawn on the final rank
    """
    if colour == WHITE:
        return piece_id == PAWN and orig[1] == '7' and dest[1] == '8'
    elif colour == BLACK:
        return piece_id == PAWN and orig[1] == '2' and dest[1] == '1'
    return False

def pieceIsStationary(chessboard, pos):
    """
    Checks if piece has not moved in the game
    """
    result = True
    # Get Starting Value
    idx = (chessboard[0][coords[pos]][0], chessboard[0][coords[pos]][1])
    # Check every position
    for board in chessboard:
        if not (board[coords[pos]][0] == idx[0] and board[coords[pos]][1] == idx[1]):
            result = False
    return result

def moveIsPossible(board, piece_id, colour, mode, orig, dest):
    """
    Checks if a move is possible, but may not be a legal move
    """
    if(piece_id == PAWN):
        return checkPawn(board, colour, mode, orig, dest)
    elif(piece_id == KNIGHT):
        return checkKnight(board, colour, mode, orig, dest)
    elif(piece_id == BISHOP):
        return checkBishop(board, colour, mode, orig, dest)
    elif(piece_id == ROOK):
        return checkRook(board, colour, mode, orig, dest)
    elif(piece_id == QUEEN):
        return checkQueen(board, colour, mode, orig, dest)
    elif(piece_id == KING):
        return checkKing(board, colour, mode, orig, dest)
    else:
        return False

def checkPawn(board, colour, mode, orig, dest):
    """
    If orig->dest is possible, return True
    """

    # The Pawn can:
    # Move 2 spaces forward if haven't moved yet
    # Move 1 space forward if not attacking (-)
    # Move forward-diagonally if attacking (x)
    
    # If there's no piece to move
    if not (checkPieceExist(board, orig, PAWN, colour)):
        return False
    
    if (colour == WHITE):
        # White + Attack
        if(mode == ATTACK):
            # If piece of opposite colour is in destination of attack
            return coords[dest][0] == coords[orig][0]-1 and abs(coords[dest][1] - coords[orig][1]) == 1 and pieceIsThere(board, dest, not WHITE)
        # White + Move:
        else:
            # If Pawn is Located on 2nd Rank
            if(coords[orig][0] == 6):
                return pathIsClear(board, orig, dest) and vertical(orig, dest) and coords[orig][0] - coords[dest][0] <= 2 and not coords[dest][0] == 7
            else:
                return pathIsClear(board, orig, dest) and vertical(orig, dest) and coords[orig][0] - coords[dest][0] == 1
    else:
        # Black + Attack
        if(mode == ATTACK):
            return coords[dest][0] == coords[orig][0]+1 and abs(coords[dest][1] - coords[orig][1]) == 1 and pieceIsThere(board, dest, not BLACK)
        # Black + Move
        else:
            if(coords[orig][0] == 1):
                return pathIsClear(board, orig, dest) and vertical(orig, dest) and coords[dest][0] - coords[orig][0] <= 2 and not coords[dest][0] == 0
            else:
                return pathIsClear(board, orig, dest) and vertical(orig, dest) and coords[dest][0] - coords[orig][0] == 1

def checkKnight(board, colour, mode, orig, dest):
    """
    If orig->dest is possible, return True
    """

    # The Knight can:
    # Move in L Shape (2 + 1)
    # Jump over pieces
    
    # If there's no piece to move
    if not (checkPieceExist(board, orig, KNIGHT, colour)):
        return False
    
    # Generate Possible Moves for the Knight
    deltas = [(-2, -1), (-2, +1), (+2, -1), (+2, +1), (-1, -2), (-1, +2), (+1, -2), (+1, +2)]
    row = coords[orig][0]
    col = coords[orig][1]
    moves = [coords_rev[(row + delta[0], col + delta[1])] for delta in deltas if 0<=row+delta[0]<8 and 0<=col+delta[1]<8]
    
    if(mode == ATTACK):
        return dest in moves and pieceIsThere(board, dest, not colour)
    else:
        return dest in moves and not anyPieceIsThere(board, dest)

def checkBishop(board, colour, mode, orig, dest):
    """
    If orig->dest is possible, return True
    """

    # The Bishop can:
    # Move Diagonally
    
    # If there's no piece to move
    if not (checkPieceExist(board, orig, BISHOP, colour)):
        return False
    
    if(mode == ATTACK):
        return not pieceInBetween(board, orig, dest) and pieceIsThere(board, dest, not colour) and diagonal(orig, dest)
    else:
        return pathIsClear(board, orig, dest) and diagonal(orig, dest)

def checkRook(board, colour, mode, orig, dest):
    """
    If orig->dest is possible, return True
    """

    # The Rook can:
    # Move Vertically and Horizontally
    
    # If there's no piece to move
    if not (checkPieceExist(board, orig, ROOK, colour)):
        return False
    
    if(mode == ATTACK):
        return not pieceInBetween(board, orig, dest) and pieceIsThere(board, dest, not colour) and (horizontal(orig, dest) or vertical(orig, dest))
    else:
        return pathIsClear(board, orig, dest) and (horizontal(orig, dest) or vertical(orig, dest))

def checkQueen(board, colour, mode, orig, dest):
    """
    If orig->dest is possible, return True
    """

    # The Queen can:
    # Move Vertically, Horizontally and Diagonally
    
    # If there's no piece to move
    if not (checkPieceExist(board, orig, QUEEN, colour)):
        return False
    
    if(mode == ATTACK):
        return not pieceInBetween(board, orig, dest) and pieceIsThere(board, dest, not colour) and (horizontal(orig, dest) or vertical(orig, dest) or diagonal(orig, dest))
    else:
        return pathIsClear(board, orig, dest) and (horizontal(orig, dest) or vertical(orig, dest) or diagonal(orig, dest))

def checkKing(board, colour, mode, orig, dest):
    """
    If orig->dest is possible, return True
    """

    # The King can:
    # Move Vertically, Horizontally and Diagonally 1 Space
    # The King cannot attack a defended piece
    
    # If there's no piece to move
    if not (checkPieceExist(board, orig, KING, colour)):
        return False
    
    deltas = [(+1, +1), (+1, 0), (+1, -1), (0, +1), (0, -1), (-1, +1), (-1, 0), (-1, -1)]
    row = coords[orig][0]
    col = coords[orig][1]
    moves = [coords_rev[(row + delta[0], col + delta[1])] for delta in deltas if 0<=row+delta[0]<8 and 0<=col+delta[1]<8]
    
    if(mode == ATTACK):
        return dest in moves and pieceIsThere(board, dest, not colour)
    else:
        return dest in moves and not anyPieceIsThere(board, dest)

def generatePossibleMoves(board, piece_id, colour, mode, orig):
    """
    Returns a list of possible moves (as strings) for a given piece
    """
    possibleMoves = []
    if(piece_id == PAWN):
        for dest in coords:
            if checkPawn(board, colour, mode, orig, dest):
                possibleMoves.append(dest)
    elif(piece_id == KNIGHT):
        for dest in coords:
            if checkKnight(board, colour, mode, orig, dest):
                possibleMoves.append(dest) 
    elif(piece_id == BISHOP):
        for dest in coords:
            if checkBishop(board, colour, mode, orig, dest):
                possibleMoves.append(dest) 
    elif(piece_id == ROOK):
        for dest in coords:
            if checkRook(board, colour, mode, orig, dest):
                possibleMoves.append(dest) 
    elif(piece_id == QUEEN):
        for dest in coords:
            if checkQueen(board, colour, mode, orig, dest):
                possibleMoves.append(dest) 
    elif(piece_id == KING):
        for dest in coords:
            if checkKing(board, colour, mode, orig, dest):
                possibleMoves.append(dest)
    return possibleMoves

def generateEveryMove(chessboard, colour, moves):
    """
    Generates every possible move for the colour player
    TODO: Edit this to contain en passant + castling
    """
    board = chessboard[-1]
    everyMove = []
    piece_list = []
    for pos in coords:
        if board[coords[pos]][0] != EMPTY and board[coords[pos]][1] == colour:
            piece_list.append((board[coords[pos]][0], pos))

    for piece, pos in piece_list:
        possibleMoves = generatePossibleMoves(board, piece, colour, MOVE, pos)
        for move in possibleMoves:
            everyMove.append((piece, pos, move, MOVE))

        # Special Move Castling
        castlingDest = []
        if colour == WHITE and pos == "e1":
            castlingDest = ["a1", "h1"]
        elif colour == BLACK and pos == "e8":
            castlingDest = ["a8", "h8"]
        for dest in castlingDest:
            if castlingIsLegal(chessboard, dest):
                everyMove.append((KING, pos, dest, MOVE))

        # Special Move En Passant
        passantDest = []
        if colour == WHITE:
            passantDest = ["a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6"]
        else:
            passantDest = ["a6", "b6", "c6", "d6", "e6", "f6", "g6", "h6"]
        for dest in passantDest:
            if enPassantIsLegal(board, moves, piece, colour, pos, dest):
                everyMove.append((PAWN, pos, dest, MOVE))

        possibleAttacks = generatePossibleMoves(board, piece, colour, ATTACK, pos)
        for att in possibleAttacks:
            everyMove.append((piece, pos, att, ATTACK))
    # print ("EVEERY MOVE")
    # print (everyMove)
    return everyMove

def generateEveryLegalMove(chessboard, colour, moves):
    """
    Generates a list of tuple every legal move for the colour player
    """
    board = chessboard[-1]
    everyMove = []
    for piece, pos, dest, mode in generateEveryMove(chessboard, colour, moves):
        if moveIsLegal(board, piece, colour, mode, pos, dest) or (piece == KING and castlingIsLegal(chessboard, dest)) or (piece == PAWN and enPassantIsLegal(board, moves, piece, colour, pos, dest)):
            everyMove.append((piece, pos, dest, mode))
    # print ("EVEry LEGAL MOVE")
    # print(everyMove)
    return everyMove

def kingInDanger(board, colour):
    """
    If King is attacked by a piece of the oppposite colour, return 1; else 0
    """
    # Locate King Position
    for pos in coords:
        if board[coords[pos]][0] == KING and board[coords[pos]][1] == colour:
            kingPos = pos
            
    # Check possibleMoves of all the pieces of opposite colour
    for pos in coords:
        if (board[coords[pos]][0] != EMPTY and board[coords[pos]][1] == (not colour)):
            if kingPos in generatePossibleMoves(board, board[coords[pos]][0], board[coords[pos]][1], ATTACK, pos):
                return True
    return False

def checkMate(chessboard, colour, moves):
    """
    If King is checkmated, return 1; else 0
    """
    return len(generateEveryLegalMove(chessboard, colour, moves)) == 0

def threefold(chessboard):
    """
    If 3 of the same positions have been made, then game is drawn
    """
    boardCheck = chessboard[-1]
    count = 0
    for board in chessboard:
        if np.array_equal(boardCheck, board):
            count += 1
    return count >= 3

def fiftyMoves(moves):
    """
    If no pawn moves and no captures for the last 50 moves, then it is drawn
    """
    result = True
    # Because a move on both sides is counted as 1 move
    if len(moves) > 100:
        last50 = moves[-100:]
        for move in last50:
            if "P" in move or "x" in move:
                result = False
    else:
        result = False
    return result

def pieceIsThere(board, pos, colour):
    """
    Returns 1 if a piece of colour is at pos on board; else return 0
    """
    return (board[coords[pos]][0] != -1 and board[coords[pos]][1] == colour)

def anyPieceIsThere(board, pos):
    """
    PieceIsThere but with any colour
    """
    return pieceIsThere(board, pos, WHITE) or pieceIsThere(board, pos, BLACK)

def checkPieceExist(board, pos, piece_id, colour):
    """
    PieceIsThere but with piece_id
    """
    return (board[coords[pos]][0] == piece_id and board[coords[pos]][1] == colour)

def pathIsClear(board, orig, dest):
    """
    If a straight line path from orig to dest is clear
    """
    return not (anyPieceIsThere(board, dest) or pieceInBetween(board, orig, dest))

def vertical(pos1, pos2):
    """
    Returns 1 if pos1 and pos2 are vertically aligned; else 0
    """
    return coords[pos1][1] == coords[pos2][1]

def horizontal(pos1, pos2):
    """
    Returns 1 if pos1 and pos2 are horizontally aligned; else 0
    """
    return coords[pos1][0] == coords[pos2][0]

def diagonal(pos1, pos2):
    """
    Returns 1 if pos1 and pos2 are diagonally aligned; else 0
    """
    return abs(coords[pos1][0] - coords[pos2][0]) == abs(coords[pos1][1] - coords[pos2][1])

def squaresInBetween(pos1, pos2):
    """
    Returns a list of strings representing the coordinates between pos1 & pos2 (non inclusive)
    """
    squares = []
    # Vertical
    if vertical(pos1, pos2):
        col = coords[pos1][1]
        if(coords[pos1][0] < coords[pos2][0]):
            start = coords[pos1][0] + 1
            end = coords[pos2][0]
        else:
            start = coords[pos2][0] + 1
            end = coords[pos1][0]
        for i in range(start, end):
            squares.append(coords_rev[(i, col)])
        return squares

    elif horizontal(pos1, pos2):
        row = coords[pos1][0]
        if(coords[pos1][1] < coords[pos2][1]):
            start = coords[pos1][1] + 1
            end = coords[pos2][1]
        else:
            start = coords[pos2][1] + 1
            end = coords[pos1][1]
        for i in range(start, end):
            squares.append(coords_rev[(row, i)])
        return squares

    # Diagonal
    elif diagonal(pos1, pos2):
        # Get a list of the possible moves from pos1
        row = coords[pos1][0]
        col = coords[pos1][1]
        
        moves = []
        moves.append([coords_rev[(row+i, col+i)] for i in range(1,8) if 0<=row+i<8 and 0<=col+i<8])
        moves.append([coords_rev[(row+i, col-i)] for i in range(1,8) if 0<=row+i<8 and 0<=col-i<8])
        moves.append([coords_rev[(row-i, col+i)] for i in range(1,8) if 0<=row-i<8 and 0<=col+i<8])
        moves.append([coords_rev[(row-i, col-i)] for i in range(1,8) if 0<=row-i<8 and 0<=col-i<8])
        
        for diag in moves:
            if pos2 in diag:
                end = diag.index(pos2)
                for i in range(end):
                    squares.append(diag[i])
        return squares
    else:
        print("Failed to check two squares")
        return []

def pieceInBetween(board, pos1, pos2):
    """
    squaresInBetween, but goes further and checks if any piece is between pos1 and pos2
    """
    val = False
    # Vertical
    if vertical(pos1, pos2):
        col = coords[pos1][1]
        if(coords[pos1][0] < coords[pos2][0]):
            start = coords[pos1][0] + 1
            end = coords[pos2][0]
        else:
            start = coords[pos2][0] + 1
            end = coords[pos1][0]
        for i in range(start, end):
            if (anyPieceIsThere(board, coords_rev[(i, col)])):
                val = True
                break
    # Horizontal
    elif horizontal(pos1, pos2):
        row = coords[pos1][0]
        if(coords[pos1][1] < coords[pos2][1]):
            start = coords[pos1][1] + 1
            end = coords[pos2][1]
        else:
            start = coords[pos2][1] + 1
            end = coords[pos1][1]
        for i in range(start, end):
            if (anyPieceIsThere(board, coords_rev[(row, i)])):
                val = True
                break
    # Diagonal
    elif diagonal(pos1, pos2):
        # Get a list of the possible moves from pos1
        row = coords[pos1][0]
        col = coords[pos1][1]
        
        moves = []
        moves.append([coords_rev[(row+i, col+i)] for i in range(1,8) if 0<=row+i<8 and 0<=col+i<8])
        moves.append([coords_rev[(row+i, col-i)] for i in range(1,8) if 0<=row+i<8 and 0<=col-i<8])
        moves.append([coords_rev[(row-i, col+i)] for i in range(1,8) if 0<=row-i<8 and 0<=col+i<8])
        moves.append([coords_rev[(row-i, col-i)] for i in range(1,8) if 0<=row-i<8 and 0<=col-i<8])
        
        for diag in moves:
            if pos2 in diag:
                end = diag.index(pos2)
                for i in range(end):
                    if (anyPieceIsThere(board, diag[i])):
                        val = True
                        break
        
    # None of the Above - Invalid Move
    else:
        val = True
    return val

def initialiseImage():
    """
    Creates Empty Board for visualisation
    """

    global game, bNewGame, bUndo, bQuit, bKnight, bBishop, bRook, bQueen, promotion

    white = np.ones((100, 100, 4)) * 0
    black = np.ones((100, 100, 4))
    black[:, :, 0:3] = 128
    black[:, :, 3] = 1
    row1 = np.concatenate([white, black, white, black, white, black, white, black], axis=1)
    row2 = np.concatenate([black, white, black, white, black, white, black, white], axis=1)
    board_vis = np.concatenate([row1, row2, row1, row2, row1, row2, row1, row2], axis=0)
    
    # Remove toolbar
    mpl.rcParams['toolbar'] = 'None'

    # Add Axis
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xticklabels(["A", "B", "C", "D", "E", "F", "G", "H"])
    ax.set_xticks([50, 150, 250, 350, 450, 550, 650, 750])
    ax.set_yticklabels(["8", "7", "6", "5", "4", "3", "2", "1"])
    ax.set_yticks([50, 150, 250, 350, 450, 550, 650, 750])

    img = plt.imshow(board_vis)

    callback = Index()

    # Add UI Buttons - New Game, Undo, 
    axNewGame = plt.axes([0.15, 0.905, 0.2, 0.075])
    bNewGame = Button(axNewGame, 'New Game')
    game = bNewGame.on_clicked(callback.bNewGame)

    axUndo = plt.axes([0.40, 0.905, 0.2, 0.075])
    bUndo = Button(axUndo, 'Undo')
    game = bUndo.on_clicked(callback.bUndo)

    axQuit = plt.axes([0.65, 0.905, 0.2, 0.075])
    bQuit = Button(axQuit, 'Quit')
    bQuit.on_clicked(callback.bQuit)

    # Add UI Buttons for Promotion
    axKnight = plt.axes([0.85, 0.7, 0.2, 0.075])
    bKnight = Button(axKnight, '', image=plt.imread(piecetoImg[(KNIGHT, WHITE)]))
    bKnight.on_clicked(callback.bKnight)

    axBishop = plt.axes([0.85, 0.5, 0.2, 0.075])
    bBishop = Button(axBishop, '', image=plt.imread(piecetoImg[(BISHOP, WHITE)]))
    bBishop.on_clicked(callback.bBishop)

    axRook = plt.axes([0.85, 0.3, 0.2, 0.075])
    bRook = Button(axRook, '', image=plt.imread(piecetoImg[(ROOK, WHITE)]))
    bRook.on_clicked(callback.bRook)

    axQueen = plt.axes([0.85, 0.1, 0.2, 0.075])
    bQueen = Button(axQueen, '', image=plt.imread(piecetoImg[(QUEEN, WHITE)]))
    bQueen.on_clicked(callback.bQueen)    

    return fig, ax, img, callback

def moveToString(piece_id, mode, orig, dest):
    if mode == MOVE:
        if piece_id == KING and orig[0] == "e" and dest[0] == "g":
            move = "0-0"
        elif piece_id == KING and orig[0] == "e" and dest[0] == "c":
            move = "0-0-0"
        else:
            move = movetoPiece_rev[piece_id] + orig + "-" + dest
    else:
        move = movetoPiece_rev[piece_id] + orig + "x" + dest
    return move

#########################################################################

###################### EVALUATION FUNCTIONS #############################
def evaluatePosition(game):
    """
    More positive -> White Winning
    More negative -> Black Winning
    0 -> Neutral
    Can be made more accurate (research later)
    """
    pieces = game.getPieces()
    score = 0
    for piece in pieces:
        if piece.colour == WHITE:
            score += pieceToScore[piece.id]
        else:
            score -= pieceToScore[piece.id]

    return score

def minimax(depth, game):
    chessboard = game.getChessboard()
    moves = game.getMoves()
    player = game.getTurn()

    if depth == 0:
        return -evaluatePosition(game)
    possibleMoves = generateEveryLegalMove(chessboard, player, moves)
    if player == WHITE:
        bestMove = -9999
        for piece_id, orig, dest, mode in possibleMoves:
            move = moveToString(piece_id, mode, orig, dest)
            game.makeMove(move)
            bestMove = max(bestMove, minimax(depth - 1, game))
            game.undoMove()
        return bestMove
    else:
        bestMove = 9999
        for piece_id, orig, dest, mode in possibleMoves:
            move = moveToString(piece_id, mode, orig, dest)
            game.makeMove(move)
            bestMove = min(bestMove, minimax(depth - 1, game))
            game.undoMove()
        return bestMove

#########################################################################

# GAME FUNCTIONS
def printMenu():
    print("n - New Game")
    print("m - Make Move")
    print("u - Undo Move")
    print("s - Print Score")
    print("p - Print Move History")
    print("q - Quit Program")

def newGame(time):
    game = ChessGame(time)
    game.resetBoard()
    return game

def move(game):
    print("Enter Move.")
    move = input()
    game.makeMove(move)
    return game

def undo(game):
    game.undoMove()
    return game

def quit():
    sys.exit()

# INTERACTIVE FUNCTIONS
class Index(object):

    def bNewGame(self, event):
        """New Game"""
        global game
        game = newGame(30)

    def bUndo(self, event):
        """Undo Move"""
        global game
        game = undo(game)

    def bQuit(self, event):
        """Quit Game (program)"""
        quit()

    def bKnight(self, event):
        """Change promotion piece to knight"""
        global promotion
        print("Promotion Piece: Knight")
        promotion = KNIGHT

    def bBishop(self, event):
        """Change promotion piece to bishop"""
        global promotion
        print("Promotion Piece: Bishop")
        promotion = BISHOP

    def bRook(self, event):
        """Change promotion piece to rook"""
        global promotion
        print("Promotion Piece: Rook")
        promotion = ROOK

    def bQueen(self, event):
        """Change promotion piece to Queen"""
        global promotion
        print("Promotion Piece: Queen")
        promotion = QUEEN

def onclick(event):
    global selection
    global game

    # Get new data
    new_selection = None
    if(event.xdata != None):
        new_selection = coords_rev[(event.ydata//100, event.xdata//100)]

    board = game.chessboard[-1]

    if new_selection != None:
        # If piece was not yet selected
        if selection == None:
            # If selected square has a piece of the right colour
            if board[coords[new_selection]][1] == game.getTurn():
                selection = new_selection
            # If empty or wrong colour
            else:
                selection = None
        # If piece is already selected
        else:
            piece_id = board[coords[selection]][0]
            orig = selection
            dest = new_selection
            valid = 0

            # If selection is a piece of the same colour
            if board[coords[new_selection]][1] == game.getTurn():
                selection = new_selection
            # If selection is an empty square
            elif board[coords[new_selection]][0] == EMPTY:
                selection = None
                move = moveToString(piece_id, MOVE, orig, dest)
                valid = game.makeMove(move)
            # If selection is a piece of the opposing colour (attack)
            else:
                selection = None
                move = moveToString(piece_id, ATTACK, orig, dest)
                valid = game.makeMove(move)
            # If makeMove was legal
            if valid and AI == COMPUTER and not game.getGameStatus():
                game.showBoard()

                piece_id, orig, dest, mode = random.choice(generateEveryLegalMove(game.chessboard, game.getTurn(), game.getMoves()))
                moveComp = moveToString(piece_id, mode, orig, dest)
                game.makeMove(moveComp)

    else:
        selection = None
    game.showBoard()

#########################################################################

############################# MAIN PROGRAM ##############################
global game

fig, ax, img, callback = initialiseImage()
fig.show()

selection = None
AI = HUMAN
# AI = COMPUTER

game = ChessGame(30)
game.resetBoard()

printMenu()

cid = fig.canvas.mpl_connect('button_press_event', lambda event: onclick(event))

action = input()
while action != 'q':
    if action == 'n':
        game = newGame(30)
    elif action == 'm':
        game = move(game)
    elif action == 'u':
        game = undo(game)
    elif action == 's':
        game.printScore()
    elif action == 'p':
        game.printMoves()

    printMenu()
    action = input()

#########################################################################
