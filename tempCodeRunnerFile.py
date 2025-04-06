def display_reminders(df: pd.DataFrame):
#     if df.empty:
#         logging.info("No reminder data available.")
#         return None, None, None
    
    
#     total_reminders = len(df)
#     acknowledged = df['Acknowledged (Yes/No)'].sum()
#     missed = total_reminders - acknowledged
    
    
#     df['Hour'] = pd.to_datetime(df['Scheduled Time'], format='%H:%M:%S').dt.hour.astype(int)
#     hourly_ack_rates = df.groupby('Hour')['Acknowledged (Yes/No)'].mean() * 100
#     best_hours = hourly_ack_rates.sort_values(ascending=False).head(3)
    
  
#     reminder_types = df['Reminder Type'].value_counts()
#     type_ack_rates = df.groupby('Reminder Type')['Acknowledged (Yes/No)'].mean() * 100
    
#     logging.info("\nReminder Statistics:")
#     logging.info(f"Total Reminders: {total_reminders}")
#     logging.info(f"Acknowledged: {acknowledged} ({acknowledged/total_reminders*100:.1f}%)")
#     logging.info(f"Missed: {missed} ({missed/total_reminders*100:.1f}%)")
    
#     logging.info("\nReminder Type Distribution:")
#     for rtype, count in reminder_types.items():
#         ack_rate = type_ack_rates[rtype]
#         logging.info(f"{rtype}: {count} reminders, {ack_rate:.1f}% acknowledged")
    
#     logging.info("\nBest Acknowledgment Hours:")
#     best_hours = hourly_ack_rates.sort_values(ascending=False).head(3)
#     for hour, rate in best_hours.items():
#         logging.info(f"{hour:02d}:00 - {rate:.1f}% acknowledged")
    
#     logging.info("\nWorst Acknowledgment Hours:")
#     worst_hours = hourly_ack_rates.sort_values().head(3)
#     for hour, rate in worst_hours.items():
#         logging.info(f"{hour:02d}:00 - {rate:.1f}% acknowledged")
    
#     logging.info("\nDaily Reminders:")
#     for _, row in df.iterrows():
#         reminder_type = row.get('Reminder Type', 'Unknown Reminder')
#         scheduled_time = row.get('Scheduled Time', 'Unknown Time')
#         user_id = row.get('Device-ID/User-ID', 'Unknown User')
#         acknowledged = row.get('Acknowledged (Yes/No)', 0)
#         status = "ACKNOWLEDGED" if acknowledged == 1 else "MISSED"
#         logging.info(f"{reminder_type} at {scheduled_time} for {user_id} => {status}")
