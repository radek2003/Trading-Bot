import torch

if torch.cuda.is_available():
    # Liczba dostępnych GPU
    num_gpus = torch.cuda.device_count()
    print(f"Liczba dostępnych GPU: {num_gpus}")

    # Lista dostępnych GPU
    for i in range(num_gpus):
        print(f"Urządzenie {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA nie jest dostępna. Brak GPU.")