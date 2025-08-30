

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import warnings
warnings.filterwarnings('ignore')

class WingAnalyzer:
    """
    A class to perform structural analysis on aircraft wings using simplified
    aerospace equations and beam theory.
    """
    
    def __init__(self):
        """Initialize the wing analyzer with default parameters."""
        self.wing_span = 0          # Wing semi-span (m)
        self.chord_length = 0       # Wing chord length (m) 
        self.total_lift = 0         # Total lift force (N)
        self.wing_weight = 0        # Wing weight (N)
        self.material_density = 0   # Material density (kg/m³)
        self.elastic_modulus = 0    # Young's modulus (Pa)
        self.safety_factor = 0      # Safety factor
        
        # Computed arrays
        self.y_positions = None     # Span-wise positions
        self.lift_distribution = None
        self.shear_force = None
        self.bending_moment = None
        self.stress_distribution = None
        self.moment_of_inertia = None
        
    def get_user_inputs(self):
        """
        Get wing parameters from user input with validation.
        """
        print("=== WING STRUCTURAL ANALYSIS TOOL ===")
        print("Enter the following wing parameters:\n")
        
        try:
            # Wing geometry
            self.wing_span = float(input("Wing semi-span (m) [default: 10]: ") or "10")
            self.chord_length = float(input("Average chord length (m) [default: 2]: ") or "2")
            
            # Load conditions
            self.total_lift = float(input("Total lift force (N) [default: 50000]: ") or "50000")
            self.wing_weight = float(input("Wing weight (N) [default: 5000]: ") or "5000")
            
            # Material properties
            print("\nMaterial Properties:")
            print("1. Aluminum (ρ=2700 kg/m³, E=70 GPa)")
            print("2. Carbon Fiber (ρ=1600 kg/m³, E=150 GPa)")
            print("3. Steel (ρ=7850 kg/m³, E=200 GPa)")
            print("4. Custom")
            
            material_choice = input("Select material [1-4, default: 1]: ") or "1"
            
            if material_choice == "1":  # Aluminum
                self.material_density = 2700
                self.elastic_modulus = 70e9
            elif material_choice == "2":  # Carbon Fiber
                self.material_density = 1600
                self.elastic_modulus = 150e9
            elif material_choice == "3":  # Steel
                self.material_density = 7850
                self.elastic_modulus = 200e9
            else:  # Custom
                self.material_density = float(input("Material density (kg/m³): "))
                self.elastic_modulus = float(input("Elastic modulus (Pa): "))
            
            self.safety_factor = float(input("Safety factor [default: 2.5]: ") or "2.5")
            
            print(f"\n=== INPUT SUMMARY ===")
            print(f"Wing semi-span: {self.wing_span:.1f} m")
            print(f"Chord length: {self.chord_length:.1f} m")
            print(f"Total lift: {self.total_lift:.0f} N")
            print(f"Wing weight: {self.wing_weight:.0f} N")
            print(f"Material density: {self.material_density:.0f} kg/m³")
            print(f"Elastic modulus: {self.elastic_modulus/1e9:.0f} GPa")
            print(f"Safety factor: {self.safety_factor:.1f}")
            
        except ValueError:
            print("Error: Please enter valid numeric values.")
            return False
        except KeyboardInterrupt:
            print("\nProgram terminated by user.")
            return False
            
        return True
    
    def calculate_lift_distribution(self, n_points=100):
        """
        Calculate lift distribution along wing span using elliptical distribution.
        
        For simplicity, we assume an elliptical lift distribution which is
        aerodynamically efficient:
        L(y) = L_max * sqrt(1 - (y/b)²)
        
        Args:
            n_points (int): Number of points along span for analysis
        """
        # Create span-wise position array
        self.y_positions = np.linspace(0, self.wing_span, n_points)
        
        # Elliptical lift distribution (Prandtl's lifting line theory)
        # L(y) = L_max * sqrt(1 - (y/b)²)
        # where L_max is determined by total lift constraint
        
        # Calculate L_max from total lift requirement
        # Total lift = ∫[0 to b] L(y) dy = L_max * (π*b/4)
        L_max = self.total_lift / (np.pi * self.wing_span / 4)
        
        # Calculate lift distribution
        normalized_y = self.y_positions / self.wing_span
        self.lift_distribution = L_max * np.sqrt(1 - normalized_y**2)
        
        # Account for wing weight (distributed uniformly)
        weight_per_unit_span = self.wing_weight / self.wing_span
        self.lift_distribution -= weight_per_unit_span
        
        print(f"Maximum lift per unit span: {L_max:.0f} N/m")
        print(f"Wing weight per unit span: {weight_per_unit_span:.0f} N/m")
    
    def calculate_shear_force(self):
        """
        Calculate shear force distribution using beam theory.
        
        Shear force V(y) = ∫[y to b] L(ξ) dξ
        where L(ξ) is the lift distribution.
        """
        dy = self.y_positions[1] - self.y_positions[0]
        self.shear_force = np.zeros_like(self.y_positions)
        
        # Integrate lift distribution from tip to root
        for i in range(len(self.y_positions)):
            # Shear force at position y is integral of lift from y to tip
            self.shear_force[i] = np.trapz(
                self.lift_distribution[i:], 
                self.y_positions[i:]
            )
    
    def calculate_bending_moment(self):
        """
        Calculate bending moment distribution using beam theory.
        
        Bending moment M(y) = ∫[y to b] V(ξ) dξ
        where V(ξ) is the shear force distribution.
        """
        self.bending_moment = np.zeros_like(self.y_positions)
        
        # Integrate shear force distribution from tip to root
        for i in range(len(self.y_positions)):
            # Bending moment at position y is integral of shear force from y to tip
            self.bending_moment[i] = np.trapz(
                self.shear_force[i:], 
                self.y_positions[i:]
            )
    
    def calculate_moment_of_inertia(self):
        """
        Calculate second moment of area for rectangular wing cross-section.
        
        For simplified analysis, assume rectangular cross-section:
        I = (chord³ * thickness) / 12
        
        Thickness estimated as 8% of chord (typical for aircraft wings)
        """
        thickness = 0.08 * self.chord_length  # 8% thickness-to-chord ratio
        self.moment_of_inertia = (self.chord_length * thickness**3) / 12
        
        print(f"Wing thickness (estimated): {thickness*1000:.1f} mm")
        print(f"Second moment of area: {self.moment_of_inertia*1e6:.2f} cm⁴")
    
    def calculate_stress_distribution(self):
        """
        Calculate bending stress distribution using flexural formula.
        
        Bending stress σ = M*c/I
        where:
        - M = bending moment
        - c = distance from neutral axis to extreme fiber (thickness/2)
        - I = second moment of area
        """
        if self.moment_of_inertia is None:
            self.calculate_moment_of_inertia()
        
        # Distance from neutral axis to extreme fiber
        thickness = 0.08 * self.chord_length
        c = thickness / 2
        
        # Calculate bending stress
        self.stress_distribution = np.abs(self.bending_moment) * c / self.moment_of_inertia
        
        # Apply safety factor
        self.stress_distribution *= self.safety_factor
        
        max_stress_mpa = np.max(self.stress_distribution) / 1e6
        print(f"Maximum bending stress (with SF={self.safety_factor}): {max_stress_mpa:.1f} MPa")
    
    def perform_analysis(self):
        """
        Perform complete wing structural analysis.
        """
        print(f"\n=== PERFORMING ANALYSIS ===")
        
        # Step 1: Calculate lift distribution
        print("1. Calculating lift distribution...")
        self.calculate_lift_distribution()
        
        # Step 2: Calculate shear force
        print("2. Calculating shear force distribution...")
        self.calculate_shear_force()
        
        # Step 3: Calculate bending moment
        print("3. Calculating bending moment distribution...")
        self.calculate_bending_moment()
        
        # Step 4: Calculate moment of inertia
        print("4. Calculating section properties...")
        self.calculate_moment_of_inertia()
        
        # Step 5: Calculate stress distribution
        print("5. Calculating stress distribution...")
        self.calculate_stress_distribution()
        
        print("Analysis complete!\n")
    
    def generate_plots(self):
        """
        Generate comprehensive plots of analysis results.
        """
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Wing Structural Analysis Results', fontsize=16, fontweight='bold')
        
        # Plot 1: Lift Distribution
        ax1.plot(self.y_positions, self.lift_distribution, 'b-', linewidth=2, label='Net Lift')
        ax1.fill_between(self.y_positions, 0, self.lift_distribution, alpha=0.3, color='blue')
        ax1.set_xlabel('Span Position (m)')
        ax1.set_ylabel('Lift per Unit Span (N/m)')
        ax1.set_title('Lift Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Shear Force Distribution
        ax2.plot(self.y_positions, self.shear_force/1000, 'r-', linewidth=2)
        ax2.fill_between(self.y_positions, 0, self.shear_force/1000, alpha=0.3, color='red')
        ax2.set_xlabel('Span Position (m)')
        ax2.set_ylabel('Shear Force (kN)')
        ax2.set_title('Shear Force Distribution')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Bending Moment Distribution
        ax3.plot(self.y_positions, self.bending_moment/1000, 'g-', linewidth=2)
        ax3.fill_between(self.y_positions, 0, self.bending_moment/1000, alpha=0.3, color='green')
        ax3.set_xlabel('Span Position (m)')
        ax3.set_ylabel('Bending Moment (kN⋅m)')
        ax3.set_title('Bending Moment Distribution')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Stress Distribution
        ax4.plot(self.y_positions, self.stress_distribution/1e6, 'm-', linewidth=2)
        ax4.fill_between(self.y_positions, 0, self.stress_distribution/1e6, alpha=0.3, color='magenta')
        ax4.set_xlabel('Span Position (m)')
        ax4.set_ylabel('Bending Stress (MPa)')
        ax4.set_title(f'Bending Stress (Safety Factor = {self.safety_factor})')
        ax4.grid(True, alpha=0.3)
        
        # Adjust layout and display
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        self.print_summary()
    
    def print_summary(self):
        """
        Print summary of analysis results.
        """
        print("=== ANALYSIS SUMMARY ===")
        print(f"Wing Configuration:")
        print(f"  Semi-span: {self.wing_span:.1f} m")
        print(f"  Chord: {self.chord_length:.1f} m")
        print(f"  Wing area: {2 * self.wing_span * self.chord_length:.1f} m²")
        print(f"  Aspect ratio: {(2 * self.wing_span) / self.chord_length:.1f}")
        
        print(f"\nLoad Conditions:")
        print(f"  Total lift: {self.total_lift/1000:.1f} kN")
        print(f"  Wing weight: {self.wing_weight/1000:.1f} kN")
        print(f"  Net upward load: {(self.total_lift - self.wing_weight)/1000:.1f} kN")
        
        print(f"\nMaximum Values:")
        print(f"  Shear force: {np.max(np.abs(self.shear_force))/1000:.1f} kN")
        print(f"  Bending moment: {np.max(np.abs(self.bending_moment))/1000:.1f} kN⋅m")
        print(f"  Bending stress: {np.max(self.stress_distribution)/1e6:.1f} MPa")
        
        # Material strength comparison (rough estimates)
        material_strengths = {
            2700: ("Aluminum", 250),    # Al 6061-T6
            1600: ("Carbon Fiber", 600), # Carbon fiber composite
            7850: ("Steel", 400)         # Steel
        }
        
        if self.material_density in material_strengths:
            material_name, yield_strength = material_strengths[self.material_density]
            max_stress_mpa = np.max(self.stress_distribution) / 1e6
            margin_of_safety = (yield_strength / max_stress_mpa) - 1
            
            print(f"\nMaterial Assessment ({material_name}):")
            print(f"  Typical yield strength: {yield_strength} MPa")
            print(f"  Maximum computed stress: {max_stress_mpa:.1f} MPa")
            print(f"  Margin of safety: {margin_of_safety:.2f}")
            
            if margin_of_safety > 0:
                print("  ✓ Design appears safe")
            else:
                print("  ⚠ WARNING: Stress exceeds material strength!")
    
    def run_analysis(self):
        """
        Main function to run the complete wing analysis.
        """
        # Get user inputs
        if not self.get_user_inputs():
            return
        
        # Perform analysis
        self.perform_analysis()
        
        # Generate plots
        print("Generating plots...")
        self.generate_plots()
        
        # Ask if user wants to save results
        save_option = input("\nSave results to file? (y/n): ").lower()
        if save_option == 'y':
            self.save_results()
    
    def save_results(self):
        """
        Save analysis results to CSV files.
        """
        try:
            # Create results array
            results = np.column_stack([
                self.y_positions,
                self.lift_distribution,
                self.shear_force,
                self.bending_moment,
                self.stress_distribution
            ])
            
            # Save to CSV
            np.savetxt('wing_analysis_results.csv', results, 
                      delimiter=',',
                      header='Span_Position_m,Lift_N_per_m,Shear_Force_N,Bending_Moment_Nm,Stress_Pa',
                      comments='')
            
            print("Results saved to 'wing_analysis_results.csv'")
            
        except Exception as e:
            print(f"Error saving results: {e}")


def main():
    """
    Main function to demonstrate wing analysis tool.
    """
    try:
        # Create analyzer instance
        analyzer = WingAnalyzer()
        
        # Run complete analysis
        analyzer.run_analysis()
        
        # Ask if user wants to run another analysis
        while True:
            run_again = input("\nRun another analysis? (y/n): ").lower()
            if run_again == 'y':
                analyzer = WingAnalyzer()
                analyzer.run_analysis()
            else:
                print("Thank you for using the Wing Structural Analysis Tool!")
                break
                
    except KeyboardInterrupt:
        print("\nProgram terminated by user. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()