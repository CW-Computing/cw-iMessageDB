SELECT
  chat.chat_identifier,
  chat.display_name,
  COUNT(message.rowid) AS message_count
FROM
  chat
JOIN
  chat_message_join ON chat.rowid = chat_message_join.chat_id
JOIN
  message ON chat_message_join.message_id = message.rowid
GROUP BY
  chat.chat_identifier, chat.display_name
ORDER BY
  message_count DESC;