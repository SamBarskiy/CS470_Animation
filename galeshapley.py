import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import numpy as np
import time

class GaleShapleyVisualizer:
    """
    Gale-Shapley Algorithm with step-by-step animation visualization.
    
    The algorithm finds a stable matching between two equal-sized sets:
    - Men (proposers) and Women (receivers)
    - Each person has a preference list ranking all members of the opposite set
    """
    
    def __init__(self, men_prefs, women_prefs, men_names=None, women_names=None):
        """
        Initialize the Gale-Shapley algorithm.
        
        Args:
            men_prefs: dict mapping man -> list of women in preference order
            women_prefs: dict mapping woman -> list of men in preference order
            men_names: optional list of men names (defaults to M0, M1, ...)
            women_names: optional list of women names (defaults to W0, W1, ...)
        """
        self.men_prefs = men_prefs
        self.women_prefs = women_prefs
        self.n = len(men_prefs)
        
        self.men = list(men_prefs.keys())
        self.women = list(women_prefs.keys())
        
        self.men_names = men_names if men_names else self.men
        self.women_names = women_names if women_names else self.women
        
        # Create woman's ranking lookup: women_rank[w][m] = rank of m for w
        self.women_rank = {}
        for w in self.women:
            self.women_rank[w] = {m: i for i, m in enumerate(women_prefs[w])}
        
        # Algorithm state - use list for deterministic ordering
        self.free_men = list(self.men)  # List to maintain proposal order
        self.engaged = {}  # woman -> man
        self.men_engaged_to = {}  # man -> woman
        self.next_proposal = {m: 0 for m in self.men}  # next woman index to propose to
        
        # History for animation
        self.history = []
        
    def run(self):
        """
        Run the Gale-Shapley algorithm and record all steps.
        Returns the stable matching.
        """
        # Initial state
        self.history.append({
            'step': 0,
            'action': 'start',
            'message': 'Algorithm starts - all men are free',
            'free_men': set(self.free_men),
            'engaged': dict(self.engaged),
            'current_man': None,
            'current_woman': None,
            'proposal_result': None
        })
        
        step = 1
        man_index = 0  # Track which man's turn it is
        
        while self.free_men:
            # Get the current man in rotation
            m = self.men[man_index]
            man_index = (man_index + 1) % self.n  # Move to next man for next iteration
            
            # Skip if this man is not free or has proposed to everyone
            if m not in self.free_men or self.next_proposal[m] >= self.n:
                # Check if any free man can still propose
                can_continue = any(
                    man in self.free_men and self.next_proposal[man] < self.n 
                    for man in self.men
                )
                if not can_continue:
                    break
                continue
                
            # Get the next woman on his preference list
            w = self.men_prefs[m][self.next_proposal[m]]
            self.next_proposal[m] += 1
            
            if w not in self.engaged:
                # Woman is free, engage them
                self.engaged[w] = m
                self.men_engaged_to[m] = w
                self.free_men.remove(m)
                
                self.history.append({
                    'step': step,
                    'action': 'engage',
                    'message': f'{m} proposes to {w} (free) → Engaged!',
                    'free_men': set(self.free_men),
                    'engaged': dict(self.engaged),
                    'current_man': m,
                    'current_woman': w,
                    'proposal_result': 'accepted'
                })
            else:
                # Woman is engaged, check if she prefers new proposer
                current_fiance = self.engaged[w]
                
                if self.women_rank[w][m] < self.women_rank[w][current_fiance]:
                    # Woman prefers new man, break old engagement
                    self.free_men.append(current_fiance)  # Add back to free list
                    del self.men_engaged_to[current_fiance]
                    
                    self.engaged[w] = m
                    self.men_engaged_to[m] = w
                    self.free_men.remove(m)
                    
                    self.history.append({
                        'step': step,
                        'action': 'replace',
                        'message': f'{m} proposes to {w} (engaged to {current_fiance}) → {w} prefers {m}, {current_fiance} is now free!',
                        'free_men': set(self.free_men),
                        'engaged': dict(self.engaged),
                        'current_man': m,
                        'current_woman': w,
                        'replaced_man': current_fiance,
                        'proposal_result': 'accepted_replaced'
                    })
                else:
                    # Woman rejects new proposer
                    self.history.append({
                        'step': step,
                        'action': 'reject',
                        'message': f'{m} proposes to {w} (engaged to {current_fiance}) → {w} rejects {m}',
                        'free_men': set(self.free_men),
                        'engaged': dict(self.engaged),
                        'current_man': m,
                        'current_woman': w,
                        'proposal_result': 'rejected'
                    })
            
            step += 1
        
        # Final state
        self.history.append({
            'step': step,
            'action': 'end',
            'message': 'Algorithm complete - stable matching found!',
            'free_men': set(self.free_men),
            'engaged': dict(self.engaged),
            'current_man': None,
            'current_woman': None,
            'proposal_result': None
        })
        
        return self.engaged
    
    def _get_detailed_explanation(self, state):
        """
        Generate a detailed explanation for each algorithm step.
        """
        action = state['action']
        
        if action == 'start':
            explanation = [
                "GALE-SHAPLEY ALGORITHM INITIALIZATION",
                "",
                "The Gale-Shapley algorithm finds a stable matching between",
                "two equal-sized groups (men and women in this example).",
                "",
                "Key Rules:",
                "1. Men propose to women in order of their preference list",
                "2. A free woman always accepts a proposal",
                "3. An engaged woman may 'trade up' if she prefers the new proposer",
                "4. A rejected man moves to the next woman on his list",
                "",
                f"Current State: All {self.n} men are FREE (green)",
                f"All {self.n} women are FREE (pink), waiting for proposals"
            ]
        
        elif action == 'engage':
            m = state['current_man']
            w = state['current_woman']
            m_pref_list = self.men_prefs[m]
            w_rank = m_pref_list.index(w) + 1
            
            explanation = [
                f"PROPOSAL: {m} proposes to {w}",
                "",
                f"{m}'s preference list: {m_pref_list}",
                f"{w} is ranked #{w_rank} on {m}'s list",
                "",
                f"CHECK: Is {w} currently engaged?",
                f"  Answer: NO - {w} is FREE",
                "",
                "RULE: A free woman always accepts a proposal",
                "",
                f"RESULT: {w} ACCEPTS!",
                f"  {m} and {w} are now ENGAGED (shown in blue)",
                f"  {m} is no longer free"
            ]
        
        elif action == 'reject':
            m = state['current_man']
            w = state['current_woman']
            current_fiance = self.engaged[w]
            w_prefs = self.women_prefs[w]
            m_rank = w_prefs.index(m) + 1
            fiance_rank = w_prefs.index(current_fiance) + 1
            
            explanation = [
                f"PROPOSAL: {m} proposes to {w}",
                "",
                f"CHECK: Is {w} currently engaged?",
                f"  Answer: YES - {w} is engaged to {current_fiance}",
                "",
                f"{w}'s preference list: {w_prefs}",
                f"  Current fiance {current_fiance} is ranked #{fiance_rank}",
                f"  New proposer {m} is ranked #{m_rank}",
                "",
                f"COMPARISON: Does {w} prefer {m} over {current_fiance}?",
                f"  #{fiance_rank} vs #{m_rank} --> Lower rank is better",
                f"  Answer: NO - {w} prefers {current_fiance} (#{fiance_rank} < #{m_rank})",
                "",
                f"RESULT: {w} REJECTS {m}!",
                f"  {m} remains free and will propose to next woman on his list"
            ]
        
        elif action == 'replace':
            m = state['current_man']
            w = state['current_woman']
            replaced = state['replaced_man']
            w_prefs = self.women_prefs[w]
            m_rank = w_prefs.index(m) + 1
            replaced_rank = w_prefs.index(replaced) + 1
            
            explanation = [
                f"PROPOSAL: {m} proposes to {w}",
                "",
                f"CHECK: Is {w} currently engaged?",
                f"  Answer: YES - {w} is engaged to {replaced}",
                "",
                f"{w}'s preference list: {w_prefs}",
                f"  Current fiance {replaced} is ranked #{replaced_rank}",
                f"  New proposer {m} is ranked #{m_rank}",
                "",
                f"COMPARISON: Does {w} prefer {m} over {replaced}?",
                f"  #{replaced_rank} vs #{m_rank} --> Lower rank is better",
                f"  Answer: YES - {w} prefers {m} (#{m_rank} < #{replaced_rank})",
                "",
                f"RESULT: {w} TRADES UP!",
                f"  {w} breaks engagement with {replaced} (now red/free)",
                f"  {w} accepts {m}'s proposal",
                f"  {replaced} goes back to proposing"
            ]
        
        elif action == 'end':
            explanation = [
                "ALGORITHM COMPLETE!",
                "",
                "A STABLE MATCHING has been found.",
                "",
                "Final Engagements:",
            ]
            for w, m in state['engaged'].items():
                explanation.append(f"  {m} <--> {w}")
            
            explanation.extend([
                "",
                "WHY IS THIS STABLE?",
                "There is no 'blocking pair' - no man and woman who",
                "would both prefer each other over their current partners.",
                "",
                "PROPERTIES:",
                "- Men-optimal: Each man gets his best achievable partner",
                "- Women-pessimal: Each woman gets her worst achievable partner",
                "- The algorithm always terminates in O(n^2) proposals"
            ])
        
        else:
            explanation = [state['message']]
        
        return explanation
    
    def visualize_interactive(self):
        """
        Create an interactive step-by-step visualization.
        Press RIGHT ARROW or SPACE to go forward, LEFT ARROW to go back.
        Press 'q' or ESC to quit.
        """
        self.current_frame = 0
        
        fig = plt.figure(figsize=(16, 10))
        
        # Create two subplots: diagram on left, explanation on right
        ax_diagram = fig.add_axes([0.02, 0.15, 0.48, 0.75])
        ax_text = fig.add_axes([0.52, 0.15, 0.46, 0.75])
        ax_nav = fig.add_axes([0.02, 0.02, 0.96, 0.10])
        
        def draw_frame():
            state = self.history[self.current_frame]
            
            # Clear axes
            ax_diagram.clear()
            ax_text.clear()
            ax_nav.clear()
            
            # ===== DIAGRAM PANEL =====
            ax_diagram.set_xlim(-0.5, 10.5)
            ax_diagram.set_ylim(-1.5, self.n + 1.5)
            ax_diagram.set_aspect('equal')
            ax_diagram.axis('off')
            ax_diagram.set_title(f"Step {state['step']}: {state['message']}", fontsize=11, pad=10)
            
            men_x = 2
            women_x = 8
            
            # Draw men
            ax_diagram.text(men_x, self.n + 0.8, 'MEN (Proposers)', ha='center', fontsize=12, fontweight='bold')
            for i, m in enumerate(self.men):
                y = self.n - i - 0.5
                
                if state['current_man'] == m:
                    color = '#FFD700'
                elif m in state['free_men']:
                    color = '#90EE90'
                else:
                    color = '#87CEEB'
                
                if state['action'] == 'replace' and state.get('replaced_man') == m:
                    color = '#FF6B6B'
                
                circle = plt.Circle((men_x, y), 0.35, color=color, ec='black', linewidth=2)
                ax_diagram.add_patch(circle)
                ax_diagram.text(men_x, y, str(m), ha='center', va='center', fontsize=10, fontweight='bold')
                
                pref_str = ', '.join(str(p) for p in self.men_prefs[m])
                ax_diagram.text(men_x - 0.6, y, f'[{pref_str}]', ha='right', va='center', fontsize=7, color='gray')
            
            # Draw women
            ax_diagram.text(women_x, self.n + 0.8, 'WOMEN (Receivers)', ha='center', fontsize=12, fontweight='bold')
            for i, w in enumerate(self.women):
                y = self.n - i - 0.5
                
                if state['current_woman'] == w:
                    color = '#FFD700'
                elif w in state['engaged']:
                    color = '#87CEEB'
                else:
                    color = '#FFB6C1'
                
                circle = plt.Circle((women_x, y), 0.35, color=color, ec='black', linewidth=2)
                ax_diagram.add_patch(circle)
                ax_diagram.text(women_x, y, str(w), ha='center', va='center', fontsize=10, fontweight='bold')
                
                pref_str = ', '.join(str(p) for p in self.women_prefs[w])
                ax_diagram.text(women_x + 0.6, y, f'[{pref_str}]', ha='left', va='center', fontsize=7, color='gray')
            
            # Draw engagement lines
            for w, m in state['engaged'].items():
                m_idx = self.men.index(m)
                w_idx = self.women.index(w)
                m_y = self.n - m_idx - 0.5
                w_y = self.n - w_idx - 0.5
                
                if state['current_man'] == m and state['current_woman'] == w and state['proposal_result'] in ['accepted', 'accepted_replaced']:
                    color = '#32CD32'
                    linewidth = 3
                else:
                    color = '#4169E1'
                    linewidth = 2
                
                ax_diagram.plot([men_x + 0.35, women_x - 0.35], [m_y, w_y], 
                               color=color, linewidth=linewidth, linestyle='-', alpha=0.7)
            
            # Draw proposal arrow
            if state['current_man'] is not None and state['current_woman'] is not None:
                m_idx = self.men.index(state['current_man'])
                w_idx = self.women.index(state['current_woman'])
                m_y = self.n - m_idx - 0.5
                w_y = self.n - w_idx - 0.5
                
                if state['proposal_result'] == 'rejected':
                    arrow_color = '#FF4444'
                else:
                    arrow_color = '#32CD32'
                
                arrow = FancyArrowPatch(
                    (men_x + 0.4, m_y), (women_x - 0.4, w_y),
                    arrowstyle='->', mutation_scale=20,
                    color=arrow_color, linewidth=2.5, linestyle='--'
                )
                ax_diagram.add_patch(arrow)
            
            # Legend
            legend_y = -1.0
            legend_items = [
                ('#90EE90', 'Free Man'),
                ('#FFB6C1', 'Free Woman'),
                ('#87CEEB', 'Engaged'),
                ('#FFD700', 'Current'),
                ('#FF6B6B', 'Replaced'),
            ]
            for i, (col, label) in enumerate(legend_items):
                x = 0.5 + i * 2
                circle = plt.Circle((x, legend_y), 0.18, color=col, ec='black')
                ax_diagram.add_patch(circle)
                ax_diagram.text(x + 0.3, legend_y, label, ha='left', va='center', fontsize=7)
            
            # ===== EXPLANATION PANEL =====
            ax_text.set_xlim(0, 1)
            ax_text.set_ylim(0, 1)
            ax_text.axis('off')
            
            # Add border
            border = patches.FancyBboxPatch(
                (0.02, 0.02), 0.96, 0.96,
                boxstyle="round,pad=0.02",
                facecolor='#F8F8F8',
                edgecolor='#333333',
                linewidth=2
            )
            ax_text.add_patch(border)
            
            ax_text.text(0.5, 0.95, "DETAILED EXPLANATION", ha='center', va='top', 
                        fontsize=13, fontweight='bold', color='#333333')
            
            explanation = self._get_detailed_explanation(state)
            
            y_start = 0.88
            line_height = 0.045
            
            for i, line in enumerate(explanation):
                y_pos = y_start - i * line_height
                if y_pos < 0.05:
                    break
                
                # Style different parts
                if line.startswith("PROPOSAL:") or line.startswith("RESULT:") or line.startswith("CHECK:") or line.startswith("COMPARISON:"):
                    color = '#0066CC'
                    weight = 'bold'
                elif line.startswith("RULE:") or line.startswith("WHY") or line.startswith("PROPERTIES:"):
                    color = '#006600'
                    weight = 'bold'
                elif "ACCEPTS" in line or "ENGAGED" in line:
                    color = '#228B22'
                    weight = 'bold'
                elif "REJECTS" in line or "TRADES UP" in line:
                    color = '#CC3300'
                    weight = 'bold'
                elif line.startswith("  "):
                    color = '#444444'
                    weight = 'normal'
                else:
                    color = '#333333'
                    weight = 'normal'
                
                ax_text.text(0.05, y_pos, line, ha='left', va='top', 
                            fontsize=9, color=color, fontweight=weight,
                            family='monospace')
            
            # ===== NAVIGATION PANEL =====
            ax_nav.set_xlim(0, 1)
            ax_nav.set_ylim(0, 1)
            ax_nav.axis('off')
            
            # Navigation background
            nav_bg = patches.FancyBboxPatch(
                (0.02, 0.1), 0.96, 0.8,
                boxstyle="round,pad=0.01",
                facecolor='#E8E8E8',
                edgecolor='#999999',
                linewidth=1
            )
            ax_nav.add_patch(nav_bg)
            
            # Progress indicator
            progress = f"Step {self.current_frame + 1} of {len(self.history)}"
            ax_nav.text(0.5, 0.5, progress, ha='center', va='center', fontsize=12, fontweight='bold')
            
            # Navigation instructions
            nav_text = "[<- LEFT] Previous    |    [RIGHT / SPACE] Next    |    [Q / ESC] Quit"
            ax_nav.text(0.5, 0.15, nav_text, ha='center', va='center', fontsize=10, color='#555555')
            
            fig.canvas.draw_idle()
        
        def on_key(event):
            if event.key in ['right', ' ', 'n']:
                if self.current_frame < len(self.history) - 1:
                    self.current_frame += 1
                    draw_frame()
            elif event.key in ['left', 'p', 'backspace']:
                if self.current_frame > 0:
                    self.current_frame -= 1
                    draw_frame()
            elif event.key in ['q', 'escape']:
                plt.close(fig)
            elif event.key == 'home':
                self.current_frame = 0
                draw_frame()
            elif event.key == 'end':
                self.current_frame = len(self.history) - 1
                draw_frame()
        
        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Initial draw
        draw_frame()
        
        plt.suptitle("GALE-SHAPLEY STABLE MATCHING ALGORITHM", fontsize=14, fontweight='bold', y=0.98)
        plt.show()
    
    def visualize(self, interval=1500, save_gif=False, filename='gale_shapley.gif'):
        """
        Create an automated animated visualization of the algorithm.
        For interactive step-by-step control, use visualize_interactive() instead.
        
        Args:
            interval: milliseconds between frames
            save_gif: whether to save as GIF
            filename: filename for the GIF
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        def draw_frame(frame_idx):
            ax.clear()
            state = self.history[frame_idx]
            
            ax.set_xlim(-0.5, 10.5)
            ax.set_ylim(-1, self.n + 2)
            ax.set_aspect('equal')
            ax.axis('off')
            ax.set_title(f"Step {state['step']}: {state['message']}", fontsize=12, pad=20)
            
            # Positions
            men_x = 1.5
            women_x = 8.5
            
            # Draw men (left side)
            ax.text(men_x, self.n + 0.8, 'Men (Proposers)', ha='center', fontsize=14, fontweight='bold')
            for i, m in enumerate(self.men):
                y = self.n - i - 0.5
                
                # Determine color
                if state['current_man'] == m:
                    color = '#FFD700'  # Gold for current proposer
                elif m in state['free_men']:
                    color = '#90EE90'  # Light green for free
                else:
                    color = '#87CEEB'  # Light blue for engaged
                
                # Check if this man was just replaced
                if state['action'] == 'replace' and state.get('replaced_man') == m:
                    color = '#FF6B6B'  # Red for just replaced
                
                circle = plt.Circle((men_x, y), 0.35, color=color, ec='black', linewidth=2)
                ax.add_patch(circle)
                ax.text(men_x, y, str(m), ha='center', va='center', fontsize=11, fontweight='bold')
                
                # Show preferences
                pref_str = ', '.join(str(p) for p in self.men_prefs[m])
                ax.text(men_x - 1.2, y, f'[{pref_str}]', ha='right', va='center', fontsize=8, color='gray')
            
            # Draw women (right side)
            ax.text(women_x, self.n + 0.8, 'Women (Receivers)', ha='center', fontsize=14, fontweight='bold')
            for i, w in enumerate(self.women):
                y = self.n - i - 0.5
                
                # Determine color
                if state['current_woman'] == w:
                    color = '#FFD700'  # Gold for current
                elif w in state['engaged']:
                    color = '#87CEEB'  # Light blue for engaged
                else:
                    color = '#FFB6C1'  # Light pink for free
                
                circle = plt.Circle((women_x, y), 0.35, color=color, ec='black', linewidth=2)
                ax.add_patch(circle)
                ax.text(women_x, y, str(w), ha='center', va='center', fontsize=11, fontweight='bold')
                
                # Show preferences
                pref_str = ', '.join(str(p) for p in self.women_prefs[w])
                ax.text(women_x + 1.2, y, f'[{pref_str}]', ha='left', va='center', fontsize=8, color='gray')
            
            # Draw engagement lines
            for w, m in state['engaged'].items():
                m_idx = self.men.index(m)
                w_idx = self.women.index(w)
                m_y = self.n - m_idx - 0.5
                w_y = self.n - w_idx - 0.5
                
                # Check if this is a new engagement in this step
                if state['current_man'] == m and state['current_woman'] == w and state['proposal_result'] in ['accepted', 'accepted_replaced']:
                    color = '#32CD32'  # Lime green for new
                    linewidth = 3
                else:
                    color = '#4169E1'  # Royal blue for existing
                    linewidth = 2
                
                ax.plot([men_x + 0.35, women_x - 0.35], [m_y, w_y], 
                       color=color, linewidth=linewidth, linestyle='-', alpha=0.7)
            
            # Draw proposal arrow for current step
            if state['current_man'] is not None and state['current_woman'] is not None:
                m_idx = self.men.index(state['current_man'])
                w_idx = self.women.index(state['current_woman'])
                m_y = self.n - m_idx - 0.5
                w_y = self.n - w_idx - 0.5
                
                if state['proposal_result'] == 'rejected':
                    arrow_color = '#FF4444'  # Red for rejection
                else:
                    arrow_color = '#32CD32'  # Green for acceptance
                
                arrow = FancyArrowPatch(
                    (men_x + 0.4, m_y), (women_x - 0.4, w_y),
                    arrowstyle='->', mutation_scale=20,
                    color=arrow_color, linewidth=2, linestyle='--'
                )
                ax.add_patch(arrow)
            
            # Legend
            legend_y = -0.5
            legend_items = [
                (plt.Circle((0, 0), 0.15, color='#90EE90', ec='black'), 'Free Man'),
                (plt.Circle((0, 0), 0.15, color='#FFB6C1', ec='black'), 'Free Woman'),
                (plt.Circle((0, 0), 0.15, color='#87CEEB', ec='black'), 'Engaged'),
                (plt.Circle((0, 0), 0.15, color='#FFD700', ec='black'), 'Current'),
                (plt.Circle((0, 0), 0.15, color='#FF6B6B', ec='black'), 'Just Replaced'),
            ]
            
            for i, (patch, label) in enumerate(legend_items):
                x = 1.5 + i * 2
                circle = plt.Circle((x, legend_y), 0.2, color=patch.get_facecolor(), ec='black')
                ax.add_patch(circle)
                ax.text(x + 0.35, legend_y, label, ha='left', va='center', fontsize=9)
            
            return []
        
        ani = animation.FuncAnimation(
            fig, draw_frame, frames=len(self.history),
            interval=interval, repeat=True, blit=False
        )
        
        if save_gif:
            ani.save(filename, writer='pillow', fps=1000/interval)
            print(f"Animation saved to {filename}")
        
        plt.tight_layout()
        plt.show()
        
        return ani
    
    def print_steps(self):
        """Print all algorithm steps to console."""
        print("=" * 60)
        print("GALE-SHAPLEY ALGORITHM EXECUTION")
        print("=" * 60)
        print(f"\nMen's preferences:")
        for m in self.men:
            print(f"  {m}: {self.men_prefs[m]}")
        print(f"\nWomen's preferences:")
        for w in self.women:
            print(f"  {w}: {self.women_prefs[w]}")
        print("\n" + "-" * 60 + "\n")
        
        for state in self.history:
            print(f"Step {state['step']}: {state['message']}")
            if state['engaged']:
                engagements = [f"{m}↔{w}" for w, m in state['engaged'].items()]
                print(f"  Engagements: {', '.join(engagements)}")
            if state['free_men']:
                print(f"  Free men: {state['free_men']}")
            print()
        
        print("=" * 60)
        print("FINAL STABLE MATCHING:")
        for w, m in self.history[-1]['engaged'].items():
            print(f"  {m} ↔ {w}")
        print("=" * 60)


def create_example_1():
    """Classic 3x3 example."""
    men_prefs = {
        'M0': ['W0', 'W1', 'W2'],
        'M1': ['W1', 'W0', 'W2'],
        'M2': ['W0', 'W1', 'W2']
    }
    women_prefs = {
        'W0': ['M1', 'M0', 'M2'],
        'W1': ['M0', 'M1', 'M2'],
        'W2': ['M0', 'M1', 'M2']
    }
    return men_prefs, women_prefs


def create_example_2():
    """4x4 example with more complex dynamics."""
    men_prefs = {
        'A': ['W', 'X', 'Y', 'Z'],
        'B': ['X', 'W', 'Y', 'Z'],
        'C': ['W', 'Y', 'X', 'Z'],
        'D': ['Y', 'W', 'X', 'Z']
    }
    women_prefs = {
        'W': ['B', 'A', 'C', 'D'],
        'X': ['A', 'B', 'C', 'D'],
        'Y': ['D', 'C', 'A', 'B'],
        'Z': ['A', 'B', 'C', 'D']
    }
    return men_prefs, women_prefs


def create_example_3():
    """Example showing replacement scenario."""
    men_prefs = {
        'M1': ['W1', 'W2', 'W3'],
        'M2': ['W1', 'W2', 'W3'],
        'M3': ['W1', 'W2', 'W3']
    }
    women_prefs = {
        'W1': ['M3', 'M2', 'M1'],
        'W2': ['M1', 'M2', 'M3'],
        'W3': ['M1', 'M2', 'M3']
    }
    return men_prefs, women_prefs


def main():
    """Main function to demonstrate the Gale-Shapley algorithm with animation."""
    print("Gale-Shapley Stable Matching Algorithm")
    print("=" * 40)
    
    # Choose example (you can switch between examples)
    print("\nRunning Example 3 (with replacements):\n")
    men_prefs, women_prefs = create_example_3()
    
    # Create visualizer and run algorithm
    gs = GaleShapleyVisualizer(men_prefs, women_prefs)
    matching = gs.run()
    
    # Print steps to console
    gs.print_steps()
    
    # Show interactive visualization
    print("\nStarting interactive visualization...")
    print("Controls:")
    print("  RIGHT ARROW / SPACE : Next step")
    print("  LEFT ARROW          : Previous step")
    print("  HOME                : Go to start")
    print("  END                 : Go to end")
    print("  Q / ESC             : Quit")
    print()
    gs.visualize_interactive()


if __name__ == "__main__":
    main()