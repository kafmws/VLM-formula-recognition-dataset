from swift.plugin import ORM, orms

from rl import syntax_reward, edit_distance_reward, render_similarity_reward

class SyntaxReward(ORM):
    def __call__(completions, **kwargs):
        return syntax_reward(completions, **kwargs)

class EditDistanceReward(ORM):
    def __call__(completions, solution, **kwargs):
        return edit_distance_reward(completions, solution, **kwargs)

class RenderSimilarityReward(ORM):
    def __call__(completions, input_image_paths, **kwargs):
        render_similarity_reward(completions, input_image_paths, **kwargs)

orms['syntax_reward']= SyntaxReward
orms['edit_distance_reward']= EditDistanceReward
orms['render_similarity_reward']= RenderSimilarityReward


