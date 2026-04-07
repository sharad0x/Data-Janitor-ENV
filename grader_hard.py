def grade(trajectory, **kwargs) -> float:
    """
    OpenEnv programmatic grader for Data Janitor - HARD task.
    Extracts the final_score from the final observation in the trajectory.
    """
    if not trajectory:
        return 0.0
        
    last_step = trajectory[-1]
    
    try:
        if isinstance(last_step, dict):
            obs = last_step.get("observation", {})
            if "final_score" in obs:
                return float(obs["final_score"])
            return float(last_step.get("reward", 0.0))
            
        if hasattr(last_step, "observation"):
            obs = last_step.observation
            if hasattr(obs, "final_score"):
                return float(obs.final_score)
                
        if hasattr(last_step, "reward"):
            return float(last_step.reward)
            
    except Exception as e:
        print(f"[Grader Error] {e}")
        
    return 0.0