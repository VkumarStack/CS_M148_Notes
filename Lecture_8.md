# Lecture 8
## Introduction
- In logistic regression, a *decision boundary* that separates the training classes in feature space is learned
- When data can be perfectly separated by a linear boundary, the data is considered **linearly separable**
- It may be the case, for linearly separable data, though, that *multiple decision boundaries can fit the data*
    - ![Linearly Separable](./Images/Linearly_Separable.png)
- The **support vector machine** algorithm tries to find the boundary that *maximizes* the margin between the two classes
    - ![svm](./Images/svm.png)
## Math Review
- Hyperplane:
    - $a_1 x_1 + a_2 x_2 + ... + a_n x_n = c$
    - $n = (a_1, a_2, ..., a_n)$
        - This is the vector that is *perpendicular* to the surface of the hyperplane
    - Given a normal $(a, b, c)$ and a point $(x_0, y_0, z_0)$:
        - $(a, b, c) \cdot (x_0 - x, y_0 - y, z_0 - z ) = 0$
    - Distance from a point $(x_0, y_0, z_0)$ to plane $ax + by + cz = d$:
        - $|(x_0 - x, y_0 - y, z_0 -z) \cdot \frac{(a, b, c)}{||(a, b, c)||}| = \frac{|ax_0 + by_0 + cz_0 - d|}{\sqrt{a^2 + b^2 + c^2}}$
        - If $(x_0, y_0, z_0)$ is on the same side as the normal vector, then $n \cdot (x_0, y_0, z_0) - d> 0$
## Classifying Linearly Separable Data
- A *linear boundary* is defined by a hyperplane: $\bold{w}^Tx + b = 0$
    - $\bold{w} = {w_1, w_2, ..., w_p}$
    - $b$ is a scalar    
- With this, classification involves determining if:
    - $w_0 + w_1 x + w_2 x_2 + ... > 0$
        - Classify $y_i = 1$
    - $w_0 + w_1 x + w_2 x_2 + ... \leq 0$
        - Classify $y_i = -1$
- The distance between any point and the distance boundary can be expressed as:
    - $D(x) = \frac{\bold{w}^Tx + b}{||\bold{w}||}$
- The points closest to the decision boundary are called hte *support vectors*
    - The margin is *twice* the distance of the boundary to its support vectors
- The goal of the support vector machine algorithm is to search for a hyperplane with the *largest margin* 
- For any plane $\bold{w}^Tx + b = 0$, scaling can be done such that the support vectors lie on the planes $\bold{w}^Tx + b = \pm1$ - this simplifies calculation for the margin as just $\frac{2}{||\bold{w}||}$
    - ![Margin](./Images/margin.png)
        - Since $\bold{w}^Tx + b = \pm 1$, the distance from each support vector is just $\frac{1}{||w||}$
- The optimization problem involves finding $\bold{w}$ and $b$ such that $p = \frac{2}{||w||}$ subject to the constraints:
    - $\bold{w}^Tx_i + b \geq 1$ if $y_i = 1$
    - $\bold{w}^Tx_i + b \leq -1$ if $y_i = -1$
- In more formal terms, the object is to find $\bold{w}$ and $b$ such that $\Phi(w) = \frac{1}{2}\bold{w}^T\bold{w}$ is minimized given the constraints:
    - For all $(x_i, y_i): y_i(\bold{w}^Tx_i + b) \geq 1$