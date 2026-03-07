import tkinter as tk
import numpy as np
from scipy.optimize import linear_sum_assignment  # pip install scipy

CELL = 100  # Size of each cell
ROW_COL_DELAY = 500
OTHER_DELAY = 2500

class HungarianAnimator:
    def __init__(self, root, matrix, click_mode=True):
        self.root = root
        self.original_matrix = matrix.astype(float)
        self.matrix = self.original_matrix.copy()
        self.n = matrix.shape[0]

        self.star = np.zeros_like(matrix, dtype=bool)
        self.prime = np.zeros_like(matrix, dtype=bool)
        self.row_cover = [False]*self.n
        self.col_cover = [False]*self.n

        self.canvas = tk.Canvas(root, width=self.n*CELL, height=self.n*CELL + 300)
        self.canvas.pack()

        self.phase = 'row_reduction'
        self.substep_row = 0
        self.substep_col = 0
        self.zero_path = []
        self.click_mode = click_mode

        self.instruction_label = "Click anywhere to advance" if click_mode else ""
        self.final_assignment = []

        self.animate_step()

    # --- Drawing ---
    def draw(self, highlight_row=None, highlight_col=None, highlight_zero=None, augment_path=None, label=None, highlight_uncovered=None, highlight_doubly=None):
        self.canvas.delete('all')
        if self.instruction_label:
            self.canvas.create_text(self.n*CELL/2, 10, text=self.instruction_label, font=('Arial', 16))
        if label:
            self.canvas.create_text(
                self.n*CELL/2, 40, 
                text=label, font=('Arial', 16),
                width=self.n*CELL-20, anchor='n'
            )

        for r in range(self.n):
            for c in range(self.n):
                x1, y1 = c*CELL, r*CELL + 200
                x2, y2 = x1 + CELL, y1 + CELL
                color = 'white'
                if highlight_row == r:
                    color = '#ffeeba'
                if highlight_col == c:
                    color = '#cce5ff'
                if self.star[r][c]:
                    color = 'lightgreen'
                elif self.prime[r][c]:
                    color = 'lightblue'
                if highlight_zero == (r, c):
                    color = 'yellow'
                if highlight_doubly and (r,c) in highlight_doubly:
                    color = '#f5b7b1'  # light red for doubly covered cells
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)
                self.canvas.create_text(x1+CELL/2, y1+CELL/2, text=str(int(round(self.matrix[r][c]))), font=('Arial', 24))

        # Row and column covers
        for r in range(self.n):
            if self.row_cover[r]:
                self.canvas.create_line(0, r*CELL+200+CELL/2, self.n*CELL, r*CELL+200+CELL/2, width=3, fill='red')
        for c in range(self.n):
            if self.col_cover[c]:
                self.canvas.create_line(c*CELL+CELL/2, 200, c*CELL+CELL/2, 200 + self.n*CELL, width=3, fill='purple')

        # Augmenting path lines
        if augment_path:
            for i in range(len(augment_path)-1):
                r1, c1 = augment_path[i]
                r2, c2 = augment_path[i+1]
                self.canvas.create_line(c1*CELL+CELL/2, r1*CELL+CELL/2+200, c2*CELL+CELL/2, r2*CELL+CELL/2+200, width=3, fill='orange')

        # Highlight uncovered cells (for adjustment)
        if highlight_uncovered:
            for r, c in highlight_uncovered:
                self.canvas.create_rectangle(c*CELL, r*CELL+200, (c+1)*CELL, (r+1)*CELL+200, outline='orange', width=3)

    # --- Row/Column operations ---
    def apply_row(self, r, m):
        self.matrix[r] -= m
        print(f'Applied row reduction: Row {r} - {m}')
        self.substep_row += 1
        self.next_step()

    def apply_col(self, c, m):
        self.matrix[:,c] -= m
        print(f'Applied column reduction: Col {c} - {m}')
        self.substep_col += 1
        self.next_step()

    # --- Hungarian core operations ---
    def star_zeros(self):
        row_has = [False]*self.n
        col_has = [False]*self.n
        for r in range(self.n):
            for c in range(self.n):
                if self.matrix[r][c]==0 and not row_has[r] and not col_has[c]:
                    self.star[r][c] = True
                    row_has[r] = True
                    col_has[c] = True
        print("Starred zeros (tentative assignments):")
        print(self.star.astype(int))

    def cover_star_columns(self):
        for c in range(self.n):
            self.col_cover[c] = any(self.star[r][c] for r in range(self.n))
        print("Columns covered due to starred zeros:", self.col_cover)

    def find_uncovered_zero(self):
        for r in range(self.n):
            if self.row_cover[r]: continue
            for c in range(self.n):
                if self.col_cover[c]: continue
                if self.matrix[r][c]==0: return (r,c)
        return None

    def build_augmenting_path(self, start_r, start_c):
        path = [(start_r, start_c)]
        done = False
        while not done:
            # Step 1: find starred zero in the same column
            r = next((r for r in range(self.n) if self.star[r][path[-1][1]]), None)
            if r is not None:
                path.append((r, path[-1][1]))
            else:
                break
            # Step 2: find primed zero in the same row
            c = next((c for c in range(self.n) if self.prime[path[-1][0]][c]), None)
            if c is not None:
                path.append((path[-1][0], c))
            else:
                done = True
        return path

    def augment_path(self, path):
        for r, c in path:
            self.star[r][c] = not self.star[r][c]
            self.prime[r][c] = False
        # After augmenting, clear all primes and covers
        self.prime.fill(False)
        self.row_cover = [False]*self.n
        self.col_cover = [False]*self.n

    # --- Adjust matrix step with animation ---
    def adjust_matrix_phase(self):
        # Collect uncovered cells (subtract min)
        uncovered = [(r, c) for r in range(self.n) for c in range(self.n)
                    if not self.row_cover[r] and not self.col_cover[c]]
        # Collect doubly covered cells (add min)
        doubly_covered = [(r, c) for r in range(self.n) for c in range(self.n)
                        if self.row_cover[r] and self.col_cover[c]]

        if not uncovered and not doubly_covered:
            print("No cells to adjust — skipping")
            self.phase = 'find_zero'
            self.next_step()
            return

        minval = min(self.matrix[r][c] for r,c in uncovered)
        
        # Draw both sets of highlights
        self.draw(
            label=f'Subtract {int(minval)} from uncovered, add {int(minval)} to doubly covered cells',
            highlight_uncovered=uncovered,
            highlight_doubly=doubly_covered
        )

        self.canvas.unbind("<Button-1>")

        def apply_adjustment():
            for r in range(self.n):
                for c in range(self.n):
                    if not self.row_cover[r] and not self.col_cover[c]:
                        self.matrix[r][c] -= minval
                    if self.row_cover[r] and self.col_cover[c]:
                        self.matrix[r][c] += minval
            print(f'Adjusted matrix by {minval} (subtract from uncovered, add to doubly covered)')
            self.phase = 'find_zero'
            self.next_step()

        self.root.after(OTHER_DELAY, apply_adjustment)

    # --- Final assignment computation ---
    def compute_final_assignment(self):
        if self.final_assignment:
            return
        row_ind, col_ind = linear_sum_assignment(self.original_matrix)
        self.final_assignment = [(int(r), int(c)) for r, c in zip(row_ind, col_ind)]
        total_cost = sum(self.original_matrix[r][c] for r,c in self.final_assignment)
        print("Final assignment (row, col):", self.final_assignment)
        print("Total minimum cost:", total_cost)
        label_lines = ["Final assignment:"]
        for r, c in self.final_assignment:
            label_lines.append(f"Row {r} → Column {c} (Cost: {int(self.original_matrix[r][c])})")
        label_lines.append(f"Total minimum cost = {int(total_cost)}")
        self.draw(label="\n".join(label_lines))

    # --- Step control ---
    def next_step(self):
        if self.click_mode:
            self.canvas.unbind("<Button-1>")
            self.canvas.bind("<Button-1>", lambda e: self.animate_step())
        else:
            delay = ROW_COL_DELAY if self.phase in ['row_reduction','col_reduction'] else OTHER_DELAY
            self.root.after(delay, self.animate_step)

    # --- Phases ---
    def cover_columns_phase(self):
        self.cover_star_columns()
        self.draw(label='Cover columns containing starred zeros.')
        # Check if assignment complete
        if np.sum(self.star) == self.n:
            print("All rows assigned → final assignment")
            self.phase = 'final_assignment'
            self.compute_final_assignment()
        else:
            self.phase = 'find_zero'
            self.next_step()

    def find_zero_phase(self):
        zero = self.find_uncovered_zero()
        if zero:
            r, c = zero
            self.prime[r][c] = True
            star_col = next((col for col in range(self.n) if self.star[r][col]), None)
            if star_col is None:
                path = self.build_augmenting_path(r, c)
                self.augment_path(path)
                self.row_cover = [False]*self.n
                self.col_cover = [False]*self.n
                self.prime.fill(False)
                self.cover_star_columns()
                label = 'Augmenting path applied to increase assignments.'
                self.draw(augment_path=path, label=label)
                self.next_step()
            else:
                self.row_cover[r] = True
                self.col_cover[star_col] = False
                label = f'Prime zero at ({r},{c}); row has starred zero. Cover row {r}, uncover column {star_col}.'
                self.draw(highlight_zero=(r,c), label=label)
                self.next_step()
        else:
            # No uncovered zeros → animate adjustment
            if np.sum(self.star) == self.n:
                self.phase = 'final_assignment'
                self.compute_final_assignment()
            else:
                self.phase = 'adjust_matrix_phase'
                self.adjust_matrix_phase()

    # --- Main animation loop ---
    def animate_step(self):
        if self.phase=='row_reduction':
            if self.substep_row < self.n:
                r = self.substep_row
                m = np.min(self.matrix[r])
                label = f'Row {r} reduction: subtract smallest value {int(m)} to create zeros.'
                self.draw(highlight_row=r, label=label)
                self.root.after(0, lambda r=r, m=m: self.apply_row(r,m))
            else:
                self.phase='col_reduction'
                self.substep_col=0
                self.next_step()
        elif self.phase=='col_reduction':
            if self.substep_col < self.n:
                c = self.substep_col
                m = np.min(self.matrix[:,c])
                label = f'Column {c} reduction: subtract smallest value {int(m)} to create zeros.'
                self.draw(highlight_col=c, label=label)
                self.root.after(0, lambda c=c, m=m: self.apply_col(c,m))
            else:
                self.phase='star_zeros'
                self.next_step()
        elif self.phase=='star_zeros':
            self.star_zeros()
            label = 'Star independent zeros (no two in same row/column) as tentative assignments.'
            self.draw(label=label)
            self.phase='cover_columns'
            self.next_step()
        elif self.phase=='cover_columns':
            self.cover_columns_phase()
        elif self.phase=='find_zero':
            self.find_zero_phase()
        elif self.phase=='adjust_matrix_phase':
            self.adjust_matrix_phase()
        elif self.phase=='final_assignment':
            return

# --- Main ---
if __name__=='__main__':
    matrix = np.array([
        [1, 2, 3, 4],
        [2, 4, 8, 7],
        [3, 6, 6, 5],
        [4, 8, 9, 10]
    ])

    root = tk.Tk()
    root.title('Hungarian Algorithm Click-to-Step Animation')
    HungarianAnimator(root, matrix, click_mode=True)
    root.mainloop()