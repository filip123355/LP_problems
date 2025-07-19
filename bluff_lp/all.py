import sys
import subprocess
import os

def set_constants(num_dices, num_faces):
    const_path = os.path.join(os.path.dirname(__file__), 'constants.py')
    with open(const_path, 'w') as f:
        f.write(f"NUM_FACES={num_faces}\nNUM_DICES={num_dices}\n")

def main():
    if len(sys.argv) != 3:
        print("Usage: python -m bluff_lp.all <num_dices> <num_faces>")
        sys.exit(1)
    num_dices = int(sys.argv[1])
    num_faces = int(sys.argv[2])
    set_constants(num_dices, num_faces)
    print(f"Set NUM_DICES={num_dices}, NUM_FACES={num_faces} in constants.py")

    # Run game_matrix.py as a module
    print("\n=== Building game matrices and constraints ===")
    subprocess.run([sys.executable, '-m', 'bluff_lp.game_matrix'], check=True)

    # Run solve.py as a module
    print("\n=== Solving game matrices ===")
    subprocess.run([sys.executable, '-m', 'bluff_lp.solve'], check=True)

    # Run test.py as a module
    print("\n=== Testing strategies ===")
    subprocess.run([sys.executable, '-m', 'bluff_lp.test'], check=True)

if __name__ == "__main__":
    main()
