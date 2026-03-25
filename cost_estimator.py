class DamageEstimator:
    def __init__(self):
        # Base industry average costs based on severity and part
        # Values are rough estimates (in USD)
        self.cost_matrix = {
            "Bumper": {
                "Minor Scratch": (150, 300),
                "Moderate Dent": (350, 500),
                "Severe Structural Damage": (600, 1000)
            },
            "Windshield": {
                "Minor Scratch": (50, 150),
                "Moderate Dent": (200, 300),     # Usually a crack for windshield
                "Severe Structural Damage": (300, 600)   # Full replacement
            },
            "Side Door": {
                "Minor Scratch": (200, 400),
                "Moderate Dent": (450, 700),
                "Severe Structural Damage": (800, 1500)
            }
        }
        
    def estimate_severity(self, pixel_area):
        """
        Maps a Damage Area (in pixels) to a rough severity level.
        (This logic assumes images are normalized to 640x640)
        """
        # A simple heuristic based on the number of damaged pixels
        if pixel_area < 5000:
            return "Minor Scratch"
        elif pixel_area < 25000:
            return "Moderate Dent"
        else:
            return "Severe Structural Damage"
            
    def get_estimate(self, part_name, mask_area_pixels):
        """
        Returns the formatted report dictionary containing part, severity, and cost range.
        """
        if part_name not in self.cost_matrix:
            return None
            
        severity = self.estimate_severity(mask_area_pixels)
        min_cost, max_cost = self.cost_matrix[part_name][severity]
        
        return {
            "Detected Part": part_name,
            "Damage Severity": severity,
            "Estimated Repair Cost": f"${min_cost} - ${max_cost}"
        }

if __name__ == "__main__":
    # Test estimation logic
    estimator = DamageEstimator()
    report = estimator.get_estimate("Bumper", 12000)
    print("Test Estimation:")
    for key, value in report.items():
        print(f"{key}: {value}")
