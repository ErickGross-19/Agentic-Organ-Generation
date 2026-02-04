"""
Skeletal tissue scaffold generators.

Provides specialized scaffold types for various skeletal tissues including:
- Trabecular bone (cancellous/spongy bone)
- Osteochondral (bone-cartilage gradient)
- Articular cartilage (zonal architecture)
- Meniscus (fibrocartilage with wedge shape)
- Tendon/ligament (aligned crimped fibers)
- Intervertebral disc (annulus fibrosus + nucleus pulposus)
- Haversian bone (cortical bone with osteons)
"""

from .trabecular_bone import (
    TrabecularBoneParams,
    generate_trabecular_bone,
    generate_trabecular_bone_from_dict
)

from .osteochondral import (
    OsteochondralParams,
    generate_osteochondral,
    generate_osteochondral_from_dict
)

from .articular_cartilage import (
    ArticularCartilageParams,
    generate_articular_cartilage,
    generate_articular_cartilage_from_dict
)

from .meniscus import (
    MeniscusParams,
    generate_meniscus,
    generate_meniscus_from_dict
)

from .tendon_ligament import (
    TendonLigamentParams,
    generate_tendon_ligament,
    generate_tendon_ligament_from_dict
)

from .intervertebral_disc import (
    IntervertebralDiscParams,
    generate_intervertebral_disc,
    generate_intervertebral_disc_from_dict
)

from .haversian_bone import (
    HaversianBoneParams,
    generate_haversian_bone,
    generate_haversian_bone_from_dict
)

__all__ = [
    # Trabecular bone
    'TrabecularBoneParams',
    'generate_trabecular_bone',
    'generate_trabecular_bone_from_dict',

    # Osteochondral
    'OsteochondralParams',
    'generate_osteochondral',
    'generate_osteochondral_from_dict',

    # Articular cartilage
    'ArticularCartilageParams',
    'generate_articular_cartilage',
    'generate_articular_cartilage_from_dict',

    # Meniscus
    'MeniscusParams',
    'generate_meniscus',
    'generate_meniscus_from_dict',

    # Tendon/ligament
    'TendonLigamentParams',
    'generate_tendon_ligament',
    'generate_tendon_ligament_from_dict',

    # Intervertebral disc
    'IntervertebralDiscParams',
    'generate_intervertebral_disc',
    'generate_intervertebral_disc_from_dict',

    # Haversian bone
    'HaversianBoneParams',
    'generate_haversian_bone',
    'generate_haversian_bone_from_dict',
]
