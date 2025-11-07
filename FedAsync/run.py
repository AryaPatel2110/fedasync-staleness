# run_async.py
from .server import AsyncFLServer

def main():
    server = AsyncFLServer(dataset_name="CIFAR10", num_clients=5)
    server.visualize_distribution()

    print("\n=== Starting async federated learning simulation ===")
    for r in range(3):
        print(f"\n--- Async Round {r+1} ---")
        server.run_async_round(epochs=1)

if __name__ == "__main__":
    main()
