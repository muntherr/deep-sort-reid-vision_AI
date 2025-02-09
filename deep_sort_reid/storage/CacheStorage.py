

from typing import Dict, List
from torch import Tensor

from deep_sort_reid.types.tracker import TrackID


class CacheStorage():
    """
    Cache of the feature samples used for re-matching tracks based on appearance.
    """

    samples: Dict[TrackID, List[Tensor]] = {}

    def __init__(self, max_samples_per_track: int):
        self.max_samples_per_track = max_samples_per_track
    

    def add_sample(self, track_id: TrackID, feature: Tensor):
        # We may want to store samples in different ways, f.e as mean features
        if track_id in self.samples:
            if len(self.samples[track_id]) > self.max_samples_per_track:
                self.samples[track_id] = self.samples[track_id][1:]

            self.samples[track_id].append(feature)

        else:
            self.samples[track_id] = [feature]

    def __getitem__(self, key):
        return self.samples[key]
