# Importamos la librería necesaria para conectarnos a MySQL
import mysql.connector


# La conexion con la base de datos

#En la variable con nombre "conexion" almacenamos la funcion de la libreria importada
#"mysql.connector.connect" que se encarga de establecer la conexión con la base de datos MySQL.

conexion = mysql.connector.connect(user='root', 
                                   password='Chato2005.', 
                                   host='localhost', 
                                   database='test', port=3306)

# Luego especificamos dentro de la función los parámetros necesarios para la conexión, como son:

# - user: usuario de MySQL
# - password: contraseña del usuario
# - host: servidor (localhost si es la máquina propia)
# - database: nombre de la base de datos
# - port: puerto por defecto de MySQL en esta oportunidad


#Este print sirve para confirmar que la conexión fue exitosa.
# Si hay error, el programa se detendrá antes de llegar aquí.
print("Conexión exitosa a la base de datos MySQL")