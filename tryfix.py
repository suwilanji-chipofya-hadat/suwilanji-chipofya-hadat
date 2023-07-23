bemba = []
english = []
for x in verses_bemba:
  for y in verses_english:
      for x_key, x_values in x.items():
        for y_key, y_values in y.items():
          b_text = ""
          e_text = ""
          if x_key == y_key:
            for val in x_values:
              b_text += val
            for val in y_values:
              e_text += val
            bemba.append(b_text)
            english.append(e_text)
