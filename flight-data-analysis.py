from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Create a Spark session
spark = SparkSession.builder.appName("Advanced Flight Data Analysis").getOrCreate()

# Load datasets
flights_df = spark.read.csv("flights.csv", header=True, inferSchema=True)
airports_df = spark.read.csv("airports.csv", header=True, inferSchema=True)
carriers_df = spark.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output paths
output_dir = "output/"
task1_output = output_dir + "task1_largest_discrepancy.csv"
task2_output = output_dir + "task2_consistent_airlines.csv"
task3_output = output_dir + "task3_canceled_routes.csv"
task4_output = output_dir + "task4_carrier_performance_time_of_day.csv"

# Ensure output directory exists
import os
os.makedirs(output_dir, exist_ok=True)

# ------------------------
# Task 1: Flights with the Largest Discrepancy Between Scheduled and Actual Travel Time
# ------------------------
def task1_largest_discrepancy(flights_df, carriers_df):
    # Calculate travel times in minutes
    flights_df = flights_df.withColumn("scheduled_time", 
                                       F.unix_timestamp("ScheduledArrival") - F.unix_timestamp("ScheduledDeparture"))
    flights_df = flights_df.withColumn("actual_time", 
                                       F.unix_timestamp("ActualArrival") - F.unix_timestamp("ActualDeparture"))
    
    # Calculate discrepancy and apply window function to rank by discrepancy
    flights_df = flights_df.withColumn("discrepancy", F.abs(F.col("scheduled_time") - F.col("actual_time")))
    window = Window.orderBy(F.desc("discrepancy"))
    
    largest_discrepancy = flights_df.withColumn("rank", F.row_number().over(window)).filter(F.col("rank") <= 10)
    largest_discrepancy.write.csv(task1_output, header=True)
    print(f"Task 1 output written to {task1_output}")

# ------------------------
# Task 2: Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def task2_consistent_airlines(flights_df, carriers_df):
    delay_stddev = flights_df.groupBy("CarrierCode").agg(
        F.count("ActualDeparture").alias("num_flights"),
        F.stddev(F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")).alias("stddev_delay")
    ).filter(F.col("num_flights") > 100)
    
    consistent_airlines = delay_stddev.join(carriers_df, "CarrierCode").orderBy("stddev_delay")
    consistent_airlines.write.csv(task2_output, header=True)
    print(f"Task 2 output written to {task2_output}")

# ------------------------
# Task 3: Origin-Destination Pairs with the Highest Percentage of Canceled Flights
# ------------------------
def task3_canceled_routes(flights_df, airports_df):
    # Add a column for canceled flights (assuming flights with null ActualArrival are canceled)
    flights_df = flights_df.withColumn("canceled", F.when(F.col("ActualArrival").isNull(), 1).otherwise(0))
    
    # Calculate cancellation rates
    canceled_routes = flights_df.groupBy("Origin", "Destination").agg(
        (F.sum("canceled") / F.count("*")).alias("cancellation_rate")
    )
    
    # Join with airport names for origin and destination
    canceled_routes = canceled_routes.join(
        airports_df.withColumnRenamed("AirportCode", "Origin")
                   .withColumnRenamed("AirportName", "Origin_Airport")
                   .withColumnRenamed("City", "Origin_City"),
        "Origin"
    ).join(
        airports_df.withColumnRenamed("AirportCode", "Destination")
                   .withColumnRenamed("AirportName", "Destination_Airport")
                   .withColumnRenamed("City", "Destination_City"),
        "Destination"
    ).orderBy(F.desc("cancellation_rate"))
    
    # Write to CSV
    canceled_routes.write.csv(task3_output, header=True)
    print(f"Task 3 output written to {task3_output}")

# ------------------------
# Task 4: Carrier Performance Based on Time of Day
# ------------------------
def task4_carrier_performance_time_of_day(flights_df, carriers_df):
    # Define time of day categories based on scheduled departure time
    flights_df = flights_df.withColumn(
        "time_of_day",
        F.when((F.hour("ScheduledDeparture") >= 5) & (F.hour("ScheduledDeparture") < 12), "Morning")
        .when((F.hour("ScheduledDeparture") >= 12) & (F.hour("ScheduledDeparture") < 17), "Afternoon")
        .when((F.hour("ScheduledDeparture") >= 17) & (F.hour("ScheduledDeparture") < 21), "Evening")
        .otherwise("Night")
    )
    
    # Calculate average delay for each carrier and time of day
    performance = flights_df.groupBy("CarrierCode", "time_of_day").agg(
        F.avg(F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")).alias("average_delay")
    ).join(carriers_df, "CarrierCode").orderBy("time_of_day", "average_delay")
    
    performance.write.csv(task4_output, header=True)
    print(f"Task 4 output written to {task4_output}")

# ------------------------
# Call the functions for each task
# ------------------------
task1_largest_discrepancy(flights_df, carriers_df)
task2_consistent_airlines(flights_df, carriers_df)
task3_canceled_routes(flights_df, airports_df)
task4_carrier_performance_time_of_day(flights_df, carriers_df)

# Stop the Spark session
spark.stop()
