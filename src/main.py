# import threading

from src.preprocess import main
from src.visualisations import main_v

# main()
main_v()

# preprocess_df = threading.Thread(target=main())
# visualisations = threading.Thread(target=main_v())

# preprocess_df.start()
# visualisations.start()