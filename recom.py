import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

class DietRecommender:
    # Class-level variables to store data
    _data = None
    _datafin = None
    _food_items = None
    _scaler = StandardScaler()
    
    @classmethod
    def _initialize(cls):
        """Initialize the class data if not already loaded"""
        if cls._data is None:
            try:
                cls._data = pd.read_csv('input.csv')
                cls._datafin = pd.read_csv('inputfin.csv')
                cls._food_items = cls._data['Food_items']
                # Replace any NaN values with 0
                cls._data = cls._data.fillna(0)
                print("Data loaded successfully")
            except Exception as e:
                print(f"Error loading data: {str(e)}")
                raise
    
    @staticmethod
    def _calculate_bmi_class(bmi):
        """Calculate BMI class and status"""
        if bmi < 16:
            return 4, "severely underweight"
        elif bmi >= 16 and bmi < 18.5:
            return 3, "underweight"
        elif bmi >= 18.5 and bmi < 25:
            return 2, "Healthy"
        elif bmi >= 25 and bmi < 30:
            return 1, "overweight"
        else:
            return 0, "severely overweight"

    @staticmethod
    def _get_age_class(age):
        """Calculate age class"""
        return round(age / 20)

    @classmethod
    def _get_nutritional_requirements(cls, bmi_class, age_class):
        """Calculate nutritional requirements based on BMI and age"""
        requirements = {
            4: {'calories': 1.4, 'protein': 1.5, 'carbs': 1.2, 'fats': 1.2},  # severely underweight
            3: {'calories': 1.2, 'protein': 1.3, 'carbs': 1.1, 'fats': 1.1},  # underweight
            2: {'calories': 1.0, 'protein': 1.0, 'carbs': 1.0, 'fats': 1.0},  # healthy
            1: {'calories': 0.8, 'protein': 1.2, 'carbs': 0.7, 'fats': 0.7},  # overweight
            0: {'calories': 0.6, 'protein': 1.3, 'carbs': 0.5, 'fats': 0.5}   # severely overweight
        }
        
        # Adjust based on age
        age_multiplier = max(0.8, min(1.2, 1 - (age_class - 2) * 0.1))
        req = requirements[bmi_class]
        return {k: v * age_multiplier for k, v in req.items()}

    @classmethod
    def _prepare_meal_features(cls, meal_data):
        """Prepare features for clustering"""
        nutritional_features = ['Calories', 'Fats', 'Proteins', 'Carbohydrates', 
                              'Fibre', 'Iron', 'Calcium', 'Sodium', 'Potassium']
        features = meal_data[nutritional_features].values
        # Replace any remaining zeros with small values to avoid division by zero
        features[features == 0] = 0.0001
        return cls._scaler.fit_transform(features)

    @classmethod
    def _cluster_foods(cls, features, n_clusters=3):
        """Perform clustering on food items"""
        if len(features) < n_clusters:
            n_clusters = max(2, len(features))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        return kmeans.fit_predict(features)

    @classmethod
    def _calculate_score(cls, food, requirements):
        """Calculate nutritional score for a food item"""
        try:
            scores = []
            
            # Avoid division by zero by adding a small epsilon
            epsilon = 0.0001
            
            # Calculate calorie score
            if food['Calories'] > epsilon:
                calorie_score = 1 - abs(food['Calories'] * requirements['calories'] - food['Calories']) / (food['Calories'] + epsilon)
                scores.append(calorie_score)
            
            # Calculate protein score
            if food['Proteins'] > epsilon:
                protein_score = 1 - abs(food['Proteins'] * requirements['protein'] - food['Proteins']) / (food['Proteins'] + epsilon)
                scores.append(protein_score)
            
            # Calculate carbs score
            if food['Carbohydrates'] > epsilon:
                carbs_score = 1 - abs(food['Carbohydrates'] * requirements['carbs'] - food['Carbohydrates']) / (food['Carbohydrates'] + epsilon)
                scores.append(carbs_score)
            
            # Calculate fat score
            if food['Fats'] > epsilon:
                fat_score = 1 - abs(food['Fats'] * requirements['fats'] - food['Fats']) / (food['Fats'] + epsilon)
                scores.append(fat_score)
            
            # Return average score, or 0 if no scores were calculated
            return np.mean(scores) if scores else 0
            
        except Exception as e:
            print(f"Error calculating score for food item: {str(e)}")
            return 0

    @classmethod
    def get_recommendations(cls, age, veg_status, weight, height):
        """Get diet recommendations based on user parameters"""
        # Initialize class data
        cls._initialize()
        
        # Calculate BMI and classes
        bmi = weight / (height ** 2)
        bmi_class, bmi_status = cls._calculate_bmi_class(bmi)
        age_class = cls._get_age_class(age)
        
        # Get nutritional requirements
        requirements = cls._get_nutritional_requirements(bmi_class, age_class)
        
        # Prepare meal masks
        meal_types = {
            'breakfast': cls._data['Breakfast'] == 1,
            'lunch': cls._data['Lunch'] == 1,
            'dinner': cls._data['Dinner'] == 1
        }
        
        recommendations = {}
        
        for meal_type, mask in meal_types.items():
            try:
                # Get meal-specific foods
                meal_foods = cls._data[mask].copy()
                
                if len(meal_foods) == 0:
                    recommendations[meal_type] = []
                    continue
                
                # Calculate scores for each food item
                scores = []
                for _, food in meal_foods.iterrows():
                    score = cls._calculate_score(food, requirements)
                    scores.append(score)
                
                meal_foods['Score'] = scores
                
                # Filter based on veg/non-veg preference
                if int(veg_status) == 0:  # vegetarian
                    meal_foods = meal_foods[meal_foods['VegNovVeg'] == 0]
                
                # Get top recommendations (at most 3, but could be fewer)
                n_recommendations = min(3, len(meal_foods))
                top_recommendations = meal_foods.nlargest(n_recommendations, 'Score')
                recommendations[meal_type] = top_recommendations['Food_items'].tolist()
                
            except Exception as e:
                print(f"Error processing {meal_type}: {str(e)}")
                recommendations[meal_type] = []
        
        return {
            'bmi': round(bmi, 2),
            'status': bmi_status,
            'recommendations': recommendations,
            'nutritional_requirements': requirements
        }

# Example usage
if __name__ == "__main__":
    try:
        # Test the recommender
        test_cases = [
            {'age': 25, 'veg_status': 1, 'weight': 70, 'height': 1.75},
            {'age': 35, 'veg_status': 0, 'weight': 85, 'height': 1.80},
            {'age': 45, 'veg_status': 1, 'weight': 60, 'height': 1.65}
        ]
        
        for case in test_cases:
            print("\nGenerating recommendations for:")
            print(f"Age: {case['age']}, Veg Status: {'Veg' if case['veg_status'] == 0 else 'Non-Veg'}")
            print(f"Weight: {case['weight']}kg, Height: {case['height']}m")
            
            recommendations = DietRecommender.get_recommendations(**case)
            
            print(f"\nBMI: {recommendations['bmi']}")
            print(f"Status: {recommendations['status']}")
            print("\nRecommended meals:")
            for meal_type, foods in recommendations['recommendations'].items():
                print(f"\n{meal_type.capitalize()}:")
                for food in foods:
                    print(f"- {food}")
                    
    except Exception as e:
        print(f"An error occurred in the main program: {str(e)}")