from manim import *
from manim.constants import *
import math

class EntropyFormula(Scene):
    def construct(self):
        BACKGROUND_COLOR = WHITE  # change this to your desired color

        # Create formula using LaTeX
        formula = MathTex("K(x_i, x_i') = exp(-\gamma \Sigma_{j=1}^p (x_{ij}-x_{i'j})^2)",font_size=50,)
        # text = Text(text='Linear Kernel').next_to(formula, DOWN)


        
            
        # Play animation to write formula
        self.play(Write(formula))
        # self.play(Write(text))

        # Wait for a moment
        self.wait(5)

       