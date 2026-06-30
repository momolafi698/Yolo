// Utilitaire de comparaison de poses pour Just Dance Captor

// Index des points clés COCO
const KP = {
  nose: 0,
  leftEye: 1, rightEye: 2,
  leftEar: 3, rightEar: 4,
  leftShoulder: 5, rightShoulder: 6,
  leftElbow: 7, rightElbow: 8,
  leftWrist: 9, rightWrist: 10,
  leftHip: 11, rightHip: 12,
  leftKnee: 13, rightKnee: 14,
  leftAnkle: 15, rightAnkle: 16
};

/**
 * Calcule l'angle formé par 3 points clés (A-B-C) avec B comme sommet.
 * Retourne l'angle en degrés [0, 180].
 */
export function calculateAngle(pA, pB, pC) {
  if (!pA || !pB || !pC || pA.score < 0.3 || pB.score < 0.3 || pC.score < 0.3) {
    return null;
  }

  const vBA = { x: pA.x - pB.x, y: pA.y - pB.y };
  const vBC = { x: pC.x - pB.x, y: pC.y - pB.y };

  const angleA = Math.atan2(vBA.y, vBA.x);
  const angleC = Math.atan2(vBC.y, vBC.x);

  let diff = Math.abs(angleA - angleC) * (180 / Math.PI);
  if (diff > 180) {
    diff = 360 - diff;
  }
  return diff;
}

/**
 * Calcule la distance euclidienne entre deux points.
 */
function distance(p1, p2) {
  return Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
}

/**
 * Calcule la largeur des épaules à l'écran pour servir de base de distance relative.
 */
export function getShoulderWidth(kps) {
  if (!kps || kps.length < 17) return 100;
  const left = kps[KP.leftShoulder];
  const right = kps[KP.rightShoulder];
  if (!left || !right || left.score < 0.15 || right.score < 0.15) return 100;
  const dist = distance(left, right);
  return dist > 10 ? dist : 100;
}

// Définitions des poses de référence (chorégraphies) par leurs angles clés cibles (en degrés)
export const DANCE_POSES = {
  // Poses pour Rasputin
  "gCzgc_RelBA": [
    {
      name: "Bras droit en l'air",
      angles: {
        rightElbow: 160,    // Coude droit tendu
        rightShoulder: 150, // Bras droit levé vers le haut
        leftElbow: 45,      // Coude gauche plié vers la hanche
        leftShoulder: 30    // Bras gauche le long du corps
      },
      check: (kps) => {
        // Poignet droit au-dessus de l'épaule et poignet gauche bas
        return kps[KP.rightWrist].y < kps[KP.rightShoulder].y && kps[KP.leftWrist].y > kps[KP.leftHip].y;
      },
      tips: {
        rightShoulder: "Lève le bras droit bien haut vers le ciel !",
        rightElbow: "Tends ton bras droit !",
        leftElbow: "Garde le bras gauche plié près du corps."
      }
    },
    {
      name: "Pose Cosiaque (Bras croisés)",
      angles: {
        rightElbow: 50,
        leftElbow: 50,
        rightShoulder: 80,
        leftShoulder: 80
      },
      check: (kps) => {
        // Poignets croisés devant la poitrine
        const sw = getShoulderWidth(kps);
        return distance(kps[KP.rightWrist], kps[KP.leftElbow]) < sw * 1.1 && 
               distance(kps[KP.leftWrist], kps[KP.rightElbow]) < sw * 1.1;
      },
      tips: {
        rightShoulder: "Garde tes coudes à hauteur de poitrine !",
        rightElbow: "Croise bien tes bras devant toi !"
      }
    },
    {
      name: "Double bras en T",
      angles: {
        rightElbow: 170,
        leftElbow: 170,
        rightShoulder: 90,
        leftShoulder: 90
      },
      check: (kps) => {
        // Poignets alignés horizontalement avec les épaules
        const sw = getShoulderWidth(kps);
        return Math.abs(kps[KP.rightWrist].y - kps[KP.rightShoulder].y) < sw * 0.8 &&
               Math.abs(kps[KP.leftWrist].y - kps[KP.leftShoulder].y) < sw * 0.8;
      },
      tips: {
        rightShoulder: "Tends tes deux bras bien à l'horizontale !",
        leftShoulder: "Tends tes deux bras bien à l'horizontale !"
      }
    }
  ],
  // Poses pour la Macarena
  "H_wZz_p20mU": [
    {
      name: "Mains sur les hanches",
      angles: {
        rightElbow: 45,
        leftElbow: 45,
        rightShoulder: 40,
        leftShoulder: 40
      },
      check: (kps) => {
        const sw = getShoulderWidth(kps);
        return distance(kps[KP.rightWrist], kps[KP.rightHip]) < sw * 0.9 &&
               distance(kps[KP.leftWrist], kps[KP.leftHip]) < sw * 0.9;
      },
      tips: {
        rightElbow: "Pose tes mains sur tes hanches et plie les coudes !",
        leftElbow: "Pose tes mains sur tes hanches et plie les coudes !"
      }
    },
    {
      name: "Bras tendus devant",
      angles: {
        rightElbow: 170,
        leftElbow: 170,
        rightShoulder: 90,
        leftShoulder: 90
      },
      check: (kps) => {
        const sw = getShoulderWidth(kps);
        // Poignets proches l'un de l'autre
        return distance(kps[KP.rightWrist], kps[KP.leftWrist]) < sw * 1.2 &&
               kps[KP.rightWrist].y < kps[KP.rightHip].y;
      },
      tips: {
        rightShoulder: "Tends tes deux bras droit devant toi à hauteur d'épaules !",
        rightElbow: "Ne plie pas les coudes !"
      }
    },
    {
      name: "Main derrière la tête",
      angles: {
        rightElbow: 30,
        rightShoulder: 120
      },
      check: (kps) => {
        const sw = getShoulderWidth(kps);
        return distance(kps[KP.rightWrist], kps[KP.rightEar]) < sw * 0.9;
      },
      tips: {
        rightShoulder: "Mets ta main droite derrière ton oreille droite !",
        rightElbow: "Plie bien le coude droit !"
      }
    }
  ],
  // Poses pour YMCA (Village People)
  "CsrffgdM450": [
    {
      name: "Lettre Y (Bras en V)",
      angles: {
        rightElbow: 170,
        leftElbow: 170,
        rightShoulder: 135,
        leftShoulder: 135
      },
      check: (kps) => {
        const sw = getShoulderWidth(kps);
        return kps[KP.rightWrist].y < kps[KP.rightShoulder].y &&
               kps[KP.leftWrist].y < kps[KP.leftShoulder].y &&
               distance(kps[KP.rightWrist], kps[KP.leftWrist]) > sw * 1.5;
      },
      tips: {
        rightShoulder: "Lève tes bras en l'air à 45 degrés pour former le Y !",
        leftShoulder: "Lève tes bras en l'air à 45 degrés pour former le Y !"
      }
    }
  ],
  // Poses pour Gangnam Style (Psy)
  "9bZkp7q19f0": [
    {
      name: "Cavalier (Poignets croisés bas)",
      angles: {
        rightElbow: 90,
        leftElbow: 90,
        rightShoulder: 45,
        leftShoulder: 45
      },
      check: (kps) => {
        const sw = getShoulderWidth(kps);
        return distance(kps[KP.rightWrist], kps[KP.leftWrist]) < sw * 0.4 &&
               kps[KP.rightWrist].y > kps[KP.rightShoulder].y;
      },
      tips: {
        rightElbow: "Plie tes bras et croise tes poignets vers le bas !",
        leftElbow: "Plie tes bras et croise tes poignets vers le bas !"
      }
    }
  ],
  // Poses pour Thriller (Michael Jackson)
  "sOnqjkJTMaA": [
    {
      name: "Zombie (Bras levés pliés)",
      angles: {
        rightElbow: 100,
        leftElbow: 100,
        rightShoulder: 80,
        leftShoulder: 80
      },
      check: (kps) => {
        return kps[KP.rightWrist].y < kps[KP.rightElbow].y &&
               kps[KP.leftWrist].y < kps[KP.leftElbow].y;
      },
      tips: {
        rightShoulder: "Lève tes bras devant toi comme un zombie !",
        rightElbow: "Plie un peu les coudes et laisse pendre tes mains !"
      }
    }
  ],
  // Poses pour Single Ladies (Beyoncé)
  "4m1EFMoRFvY": [
    {
      name: "Main gauche sur la tête",
      angles: {
        leftElbow: 30,
        leftShoulder: 120,
        rightElbow: 45,
        rightShoulder: 40
      },
      check: (kps) => {
        const sw = getShoulderWidth(kps);
        return distance(kps[KP.leftWrist], kps[KP.leftEar]) < sw * 0.8;
      },
      tips: {
        leftShoulder: "Lève la main gauche près de ta tempe !",
        rightElbow: "Pose ta main droite sur ta hanche !"
      }
    }
  ],
  // Poses pour Waka Waka (Shakira)
  "pRpeEdMmmQ0": [
    {
      name: "Mains jointes (Prière)",
      angles: {
        rightElbow: 60,
        leftElbow: 60,
        rightShoulder: 70,
        leftShoulder: 70
      },
      check: (kps) => {
        const sw = getShoulderWidth(kps);
        return distance(kps[KP.rightWrist], kps[KP.leftWrist]) < sw * 0.3 &&
               kps[KP.rightWrist].y < kps[KP.rightHip].y &&
               kps[KP.rightWrist].y > kps[KP.rightShoulder].y;
      },
      tips: {
        rightElbow: "Joins tes deux mains devant ta poitrine !",
        leftElbow: "Joins tes deux mains devant ta poitrine !"
      }
    }
  ],
  // Poses pour Disco Fever (John Travolta)
  "fNFzfwLM72c": [
    {
      name: "Pointé Disco",
      angles: {
        rightElbow: 170,
        rightShoulder: 135,
        leftElbow: 170,
        leftShoulder: 25
      },
      check: (kps) => {
        return kps[KP.rightWrist].y < kps[KP.rightShoulder].y &&
               kps[KP.leftWrist].y > kps[KP.leftHip].y;
      },
      tips: {
        rightShoulder: "Tends ton bras droit en diagonale vers le ciel !",
        leftShoulder: "Pointe ton bras gauche vers le bas !"
      }
    }
  ],
  // Poses pour Asereje (The Ketchup Song)
  "V0PisGe66mY": [
    {
      name: "Mains croisées devant le front",
      angles: {
        rightElbow: 45,
        leftElbow: 45,
        rightShoulder: 110,
        leftShoulder: 110
      },
      check: (kps) => {
        const sw = getShoulderWidth(kps);
        return distance(kps[KP.rightWrist], kps[KP.nose]) < sw * 0.8 &&
               distance(kps[KP.leftWrist], kps[KP.nose]) < sw * 0.8;
      },
      tips: {
        rightShoulder: "Lève tes mains au niveau de ton visage !",
        rightElbow: "Croise tes mains près du front !"
      }
    }
  ],
  // Poses pour Never Gonna Give You Up (Rick Astley)
  "dQw4w9WgXcQ": [
    {
      name: "Bras droit levé",
      angles: {
        rightElbow: 160,
        rightShoulder: 150,
        leftElbow: 45,
        leftShoulder: 30
      },
      check: (kps) => {
        return kps[KP.rightWrist].y < kps[KP.rightShoulder].y && kps[KP.leftWrist].y > kps[KP.leftHip].y;
      },
      tips: {
        rightShoulder: "Lève ton bras droit bien haut !",
        leftElbow: "Plie le bras gauche près de la hanche."
      }
    }
  ]
};

/**
 * Compare la pose de l'utilisateur à la chorégraphie en cours.
 * Identifie la pose la plus proche et retourne le score, la précision et le feedback.
 */
export function evaluateUserPose(kps, referenceVideoId) {
  // S'assurer que le squelette est visible
  if (!kps || kps.length < 17) {
    return {
      scoreGained: 0,
      precision: 0,
      rating: { text: "EN ATTENTE", color: "text-slate-500" },
      comment: "Placez-vous entièrement face à la caméra."
    };
  }

  // Vérifier si des poses de référence existent pour cette vidéo
  const targetPoses = DANCE_POSES[referenceVideoId] || DANCE_POSES["gCzgc_RelBA"]; // Repli par défaut
  
  let bestPoseMatch = null;
  let highestSimilarity = 0;
  let detectedTips = [];

  // 1. Calculer les angles de l'utilisateur pour les comparer
  const userAngles = {
    rightElbow: calculateAngle(kps[KP.rightShoulder], kps[KP.rightElbow], kps[KP.rightWrist]),
    leftElbow: calculateAngle(kps[KP.leftShoulder], kps[KP.leftElbow], kps[KP.leftWrist]),
    rightShoulder: calculateAngle(kps[KP.rightHip], kps[KP.rightShoulder], kps[KP.rightElbow]),
    leftShoulder: calculateAngle(kps[KP.leftHip], kps[KP.leftShoulder], kps[KP.leftElbow]),
    rightKnee: calculateAngle(kps[KP.rightHip], kps[KP.rightKnee], kps[KP.rightAnkle]),
    leftKnee: calculateAngle(kps[KP.leftHip], kps[KP.leftKnee], kps[KP.leftAnkle])
  };

  // 2. Parcourir les poses cibles pour trouver celle qui correspond le mieux
  for (const pose of targetPoses) {
    let angleDiffSum = 0;
    let validAnglesCount = 0;
    const tipsList = [];

    // Comparer les angles spécifiés dans la pose cible
    for (const [joint, targetAngle] of Object.entries(pose.angles)) {
      const userAngle = userAngles[joint];
      if (userAngle !== null && userAngle !== undefined) {
        const diff = Math.abs(userAngle - targetAngle);
        angleDiffSum += diff;
        validAnglesCount++;

        // Si l'écart sur cette articulation est grand, générer un conseil
        if (diff > 40 && pose.tips[joint]) {
          tipsList.push(pose.tips[joint]);
        }
      }
    }

    if (validAnglesCount > 0) {
      const averageDiff = angleDiffSum / validAnglesCount;
      // Convertir la différence d'angles en pourcentage de précision (écart max toléré 60°)
      const precision = Math.max(0, Math.min(100, Math.round(100 - (averageDiff / 65) * 100)));

      // Si l'utilisateur effectue le "check" logique de la pose, on booste la confiance de détection
      const matchesLogic = pose.check ? pose.check(kps) : true;
      const finalPrecision = matchesLogic ? precision : Math.max(0, precision - 25);

      if (finalPrecision > highestSimilarity) {
        highestSimilarity = finalPrecision;
        bestPoseMatch = pose;
        detectedTips = tipsList;
      }
    }
  }

  // 3. Produire le feedback et les scores si une pose a été reconnue
  if (highestSimilarity > 45 && bestPoseMatch) {
    let scoreGained = 0;
    let rating = { text: "RATÉ", color: "text-red-500" };
    let comment = detectedTips[0] || `Continue ainsi sur la pose "${bestPoseMatch.name}" !`;

    if (highestSimilarity >= 80) {
      scoreGained = 150;
      rating = { text: "PARFAIT !", color: "text-emerald-400 font-extrabold" };
      comment = `Félicitations ! Pose "${bestPoseMatch.name}" exécutée avec brio.`;
    } else if (highestSimilarity >= 60) {
      scoreGained = 80;
      rating = { text: "PRESQUE", color: "text-amber-400" };
    } else {
      scoreGained = 10;
      rating = { text: "RATÉ", color: "text-red-500" };
      comment = comment || "Regarde bien la vidéo de référence pour t'ajuster !";
    }

    return {
      scoreGained,
      precision: highestSimilarity,
      rating,
      comment
    };
  }

  // Cas où l'utilisateur ne fait aucune des poses cibles
  return {
    scoreGained: 0,
    precision: 0,
    rating: { text: "EN ATTENTE", color: "text-slate-500" },
    comment: "Imitez les mouvements de la vidéo pour gagner des points !"
  };
}

/**
 * Analyse la pose en direct pour identifier automatiquement quel style ou chorégraphie
 * l'utilisateur est en train d'exécuter.
 */
export function detectDanceStyle(kps) {
  if (!kps || kps.length < 17) return null;

  // Calculer les angles de l'utilisateur
  const rightElbow = calculateAngle(kps[KP.rightShoulder], kps[KP.rightElbow], kps[KP.rightWrist]);
  const leftElbow = calculateAngle(kps[KP.leftShoulder], kps[KP.leftElbow], kps[KP.leftWrist]);
  const rightShoulder = calculateAngle(kps[KP.rightHip], kps[KP.rightShoulder], kps[KP.rightElbow]);
  const leftShoulder = calculateAngle(kps[KP.leftHip], kps[KP.leftShoulder], kps[KP.leftElbow]);

  // Seuil de détection d'angle
  if (rightElbow === null || leftElbow === null || rightShoulder === null || leftShoulder === null) {
    return null;
  }

  const sw = getShoulderWidth(kps);

  // 1. Détection de Rasputin (Bras croisés)
  const distRightWristLeftElbow = distance(kps[KP.rightWrist], kps[KP.leftElbow]);
  const distLeftWristRightElbow = distance(kps[KP.leftWrist], kps[KP.rightElbow]);
  if (distRightWristLeftElbow < sw * 1.15 && distLeftWristRightElbow < sw * 1.15 && rightShoulder > 50 && leftShoulder > 50) {
    return "Rasputin";
  }

  // 2. Détection de la Macarena (Mains sur les hanches)
  const distRightWristRightHip = distance(kps[KP.rightWrist], kps[KP.rightHip]);
  const distLeftWristLeftHip = distance(kps[KP.leftWrist], kps[KP.leftHip]);
  if (distRightWristRightHip < sw * 1.0 && distLeftWristLeftHip < sw * 1.0 && rightElbow < 70 && leftElbow < 70) {
    return "Macarena";
  }

  // 3. Détection de YMCA (Bras en V levés)
  if (kps[KP.rightWrist].y < kps[KP.rightShoulder].y && kps[KP.leftWrist].y < kps[KP.leftShoulder].y &&
      rightShoulder > 110 && leftShoulder > 110 && distance(kps[KP.rightWrist], kps[KP.leftWrist]) > sw * 1.3) {
    return "YMCA";
  }

  // 4. Détection de Gangnam Style (Poignets croisés bas)
  if (distance(kps[KP.rightWrist], kps[KP.leftWrist]) < sw * 0.5 &&
      kps[KP.rightWrist].y > kps[KP.rightShoulder].y &&
      rightElbow < 110 && leftElbow < 110 && rightShoulder < 60 && leftShoulder < 60) {
    return "Gangnam Style";
  }

  // 5. Détection de Thriller (Pose Zombie)
  if (kps[KP.rightWrist].y < kps[KP.rightElbow].y && kps[KP.leftWrist].y < kps[KP.leftElbow].y &&
      rightShoulder > 60 && leftShoulder > 60 && rightElbow < 120 && leftElbow < 120) {
    return "Thriller";
  }

  // 6. Détection de Single Ladies (Main gauche à l'oreille, main droite sur la hanche)
  if (distance(kps[KP.leftWrist], kps[KP.leftEar]) < sw * 0.9 &&
      distance(kps[KP.rightWrist], kps[KP.rightHip]) < sw * 1.0) {
    return "Single Ladies";
  }

  // 7. Détection de Waka Waka (Mains jointes poitrine)
  if (distance(kps[KP.rightWrist], kps[KP.leftWrist]) < sw * 0.4 &&
      kps[KP.rightWrist].y > kps[KP.rightShoulder].y && kps[KP.rightWrist].y < kps[KP.rightHip].y) {
    return "Waka Waka";
  }

  // 8. Détection de Disco Fever (John Travolta)
  if (kps[KP.rightWrist].y < kps[KP.rightShoulder].y && kps[KP.leftWrist].y > kps[KP.leftHip].y &&
      rightShoulder > 110 && leftShoulder < 45) {
    return "Disco Fever";
  }

  // 9. Détection de Asereje (Mains près du front)
  if (distance(kps[KP.rightWrist], kps[KP.nose]) < sw * 0.9 && distance(kps[KP.leftWrist], kps[KP.nose]) < sw * 0.9) {
    return "Asereje";
  }

  // 10. Détection de Rickroll / Never Gonna Give You Up (Bras droit levé, coude gauche plié)
  if (kps[KP.rightWrist].y < kps[KP.rightShoulder].y && rightShoulder > 120 && leftElbow < 75 && kps[KP.leftWrist].y > kps[KP.leftHip].y) {
    return "Never Gonna Give You Up";
  }

  return null;
}

/**
 * Recherche instantanément une vidéo YouTube correspondant à la danse détectée
 * en interrogeant des instances d'API web publiques sans CORS.
 * Retourne le videoId de la première vidéo trouvée ou null.
 */
export async function searchDanceVideoOnWeb(danceName, platform = "youtube") {
  const queryText = platform === "tiktok" 
    ? `TikTok ${danceName} trend dance` 
    : `Just Dance ${danceName} gameplay`;
    
  const query = encodeURIComponent(queryText);
  
  // Liste d'instances Invidious publiques (miroirs YouTube avec API JSON ouverte et pas de CORS restrictif)
  const instances = [
    "https://invidious.projectsegfau.lt",
    "https://invidious.lunar.icu",
    "https://yewtu.be",
    "https://invidious.nerdvpn.de"
  ];

  for (const instance of instances) {
    try {
      const url = `${instance}/api/v1/search?q=${query}&type=video`;
      const response = await fetch(url);
      if (response.ok) {
        const data = await response.json();
        if (data && data.length > 0) {
          // Prendre le premier résultat vidéo trouvé
          const video = data.find(item => item.type === "video" || item.videoId);
          if (video && video.videoId) {
            console.log(`[Web Search] Vidéo trouvée sur l'instance ${instance} pour ${danceName} (${platform}) : ${video.videoId}`);
            return {
              videoId: video.videoId,
              title: video.title,
              instanceUsed: instance
            };
          }
        }
      }
    } catch (e) {
      console.warn(`[Web Search] L'instance ${instance} a échoué :`, e);
      // Continuer vers l'instance suivante en cas d'erreur
    }
  }

  // Fallback local si la recherche en ligne a complètement échoué
  console.warn("[Web Search] Échec de la recherche en ligne. Utilisation du fallback local.");
  const localFallbacks = {
    "youtube": {
      "Rasputin": { videoId: "gCzgc_RelBA", title: "Rasputin - Boney M (Just Dance)" },
      "Macarena": { videoId: "H_wZz_p20mU", title: "Macarena - Los Del Rio (Just Dance)" },
      "Never Gonna Give You Up": { videoId: "dQw4w9WgXcQ", title: "Rick Astley - Never Gonna Give You Up" },
      "YMCA": { videoId: "CsrffgdM450", title: "YMCA - Village People (Just Dance)" },
      "Gangnam Style": { videoId: "9bZkp7q19f0", title: "PSY - GANGNAM STYLE (Just Dance)" },
      "Thriller": { videoId: "sOnqjkJTMaA", title: "Michael Jackson - Thriller (Just Dance)" },
      "Single Ladies": { videoId: "4m1EFMoRFvY", title: "Beyoncé - Single Ladies (Just Dance)" },
      "Waka Waka": { videoId: "pRpeEdMmmQ0", title: "Shakira - Waka Waka (Just Dance)" },
      "Disco Fever": { videoId: "fNFzfwLM72c", title: "Disco Fever - Just Dance Gameplay" },
      "Asereje": { videoId: "V0PisGe66mY", title: "Asereje (The Ketchup Song) - Just Dance" }
    },
    "tiktok": {
      "Rasputin": { videoId: "bO2tq4zXG3A", title: "Rasputin TikTok Dance Trend" },
      "Macarena": { videoId: "G0eH9n64Uts", title: "Macarena TikTok Dance Challenge" },
      "Never Gonna Give You Up": { videoId: "gB4D914Qooc", title: "Rickroll TikTok compilation" },
      "YMCA": { videoId: "d6_cO05_0dM", title: "YMCA TikTok Dance Challenge" },
      "Gangnam Style": { videoId: "p1Gf7g3216g", title: "Gangnam Style TikTok Challenge" },
      "Thriller": { videoId: "2nB1n34927b", title: "Thriller TikTok Dance Trend" },
      "Single Ladies": { videoId: "g8B3216b9bc", title: "Single Ladies TikTok Dance" },
      "Waka Waka": { videoId: "pRm2b87f62c", title: "Waka Waka Shakira TikTok Trend" },
      "Disco Fever": { videoId: "d7B0n631a0b", title: "Disco Dance TikTok Trend" },
      "Asereje": { videoId: "fG831a89c20", title: "Asereje Ketchup Song TikTok Challenge" }
    }
  };
  
  const platformFallbacks = localFallbacks[platform] || localFallbacks["youtube"];
  return platformFallbacks[danceName] || null;
}


