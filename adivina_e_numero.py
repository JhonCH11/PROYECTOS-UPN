import random 

print("Adivina el número  (1 al 10) ")

jugar = True
while jugar:
    numero_secreto = random.randint(1, 10)
    intentos = 0
    
    while True:
        intento = int(input("Ingresa un número: "))
        
        intentos += 1

        if intento < numero_secreto:
            print("Número muy bajo, sigue intentando")
        elif intento > numero_secreto:
            print("Número muy alto, sigue intentando")
        else:
            print(f"Correcto, ¡lo lograste en {intentos} intentos!")
            break
    
    opcion = input("¿Quieres jugar de nuevo? (si / no): ").lower()
    if opcion != 'si':
        jugar = False
        print("¡Gracias por jugar!")
        