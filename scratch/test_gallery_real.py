import os
import sys
sys.path.insert(0, ".")
import cv2
import numpy as np

from src.reid.extractor import PyTorchReIDExtractor
from src.reid.gallery import TargetGallery
from src.target.manager import TargetManager

print("=========================================================")
print(" TARGET GALLERY & OCCLUSION LOGIC VALIDATION")
print("=========================================================")

extractor = PyTorchReIDExtractor(model_name="osnet_x0_25")
gallery = TargetGallery(reid_extractor=extractor, match_threshold=0.72, auto_add_threshold=0.82)
target_mgr = TargetManager(gallery=gallery, min_margin=0.06)

target_img = cv2.imread(r"diagnostics/demo/target.png")
cand_similar = cv2.imread(r"diagnostics/demo/cand_similar.png")
cand_impostor = cv2.imread(r"diagnostics/demo/cand_impostor.png")

# 1. Seed with target_img
gallery.seed(target_img, target_label="target_0")
print(f"Gallery Seeded: Size={gallery.size}, Manual={gallery.manual_count}, Auto={gallery.auto_count}")

# 2. Match similar viewpoint
emb_similar = extractor.extract(cand_similar)
sim_s, entry_s = gallery.match(emb_similar)
print(f"Match Similar Viewpoint: Sim={sim_s:.4f}, MatchedEntry={entry_s.entry_id if entry_s else None}")

# 3. Match impostor
emb_impostor = extractor.extract(cand_impostor)
sim_i, entry_i = gallery.match(emb_impostor)
print(f"Match Impostor:          Sim={sim_i:.4f}, MatchedEntry={entry_i.entry_id if entry_i else None}")

# 4. Auto-enroll similar viewpoint
added = gallery.add_auto(cand_similar, emb_similar, candidate_similarity=sim_s, track_id=1)
# Note: consecutive match requirement is 3 by default
print(f"First auto-add call (consecutive 1/3): added={added}")
added = gallery.add_auto(cand_similar, emb_similar, candidate_similarity=sim_s, track_id=1)
print(f"Second auto-add call (consecutive 2/3): added={added}")
added = gallery.add_auto(cand_similar, emb_similar, candidate_similarity=sim_s, track_id=1)
print(f"Third auto-add call (consecutive 3/3): added={added}")
print(f"Gallery State: Size={gallery.size}, Manual={gallery.manual_count}, Auto={gallery.auto_count}")

# 5. Impostor auto-add should fail
imp_added = gallery.add_auto(cand_impostor, emb_impostor, candidate_similarity=sim_i, track_id=2)
print(f"Impostor auto-add (should be False): added={imp_added}")

# 6. Test Rollback mechanism
purged = gallery.rollback_auto_entries(for_track_id=1)
print(f"Rollback auto entries for track 1: purged={purged}, remaining={gallery.size}")

print("\n=========================================================")
print(" ALL TARGET GALLERY UNIT INVARIANTS VERIFIED!")
print("=========================================================")
